"""Threaded audio facade shared by the desktop UI and future EQ DSP."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread

import numpy as np
import sounddevice as sd
from scipy.signal import lfilter

from spectrum_analyzer.audio import AudioCapture
from spectrum_analyzer.fft import SpectrumAnalyzer

BANDS = ("Bass", "Low Mid", "Mid", "High Mid", "Treble")
AUDIO_BLOCK_SIZE = 2048
GENRE_PROFILES = {
    "Flat": (0.0, 0.0, 0.0, 0.0, 0.0),
    "Pop": (2.0, 0.5, 0.0, 1.0, 1.5),
    "Turkish Pop": (2.5, 1.0, 0.0, 1.5, 1.5),
    "Rock": (3.0, 1.0, -0.5, 1.5, 2.0),
    "Metal": (4.0, 1.5, -1.0, 2.0, 3.0),
    "Jazz": (1.5, 1.0, 1.0, 1.5, 1.0),
    "Classical": (1.0, 0.0, 1.0, 1.0, 1.0),
    "Hip-Hop": (5.0, 1.0, 0.0, 0.5, 1.0),
    "Electronic": (4.0, 1.0, 0.0, 2.0, 3.0),
    "Podcast": (-2.0, 1.5, 3.0, 2.0, -1.0),
}
PRESETS = {
    "Flat": (0.0, 0.0, 0.0, 0.0, 0.0),
    "Music": (2.0, 1.0, 0.0, 1.5, 2.5),
    "Bass Boost": (7.0, 3.0, 0.0, -1.0, 1.0),
    "Vocal": (-1.0, 1.0, 3.0, 2.0, 0.0),
    "Gaming": (3.0, 1.0, -1.0, 2.0, 3.0),
    "Movie": (4.0, 1.0, 0.0, 1.0, 4.0),
}


@dataclass(frozen=True)
class EngineSnapshot:
    active: bool
    enabled: bool
    mode: str
    gains: tuple[float, ...]
    spectrum: tuple[np.ndarray, np.ndarray] | None
    sample_rate: float
    latency_ms: float
    input_name: str
    output_name: str
    error: str | None


class AudioEngine:
    """Keep capture/DSP work off the Qt thread and expose immutable snapshots."""

    def __init__(self, device_index: int | None = None, fft_size: int = 2048) -> None:
        devices = AudioCapture.list_devices()
        output_devices = AudioCapture.list_output_devices()
        if not devices:
            raise RuntimeError(
                "VB-CABLE bulunamadı. VB-Audio VB-CABLE kurun ve Windows ses çıkışını "
                "CABLE Input olarak seçin. Fiziksel cihaz loopback'i kullanılmayacak."
            )
        if not output_devices:
            raise RuntimeError("Kullanılabilir Windows çıkış cihazı bulunamadı.")
        self.devices = devices
        self.output_devices = output_devices
        self.device_index = devices[0][0]
        self.output_device_index = self._preferred_output_device()
        self.left_output_device_index = self.output_device_index
        self.right_output_device_index = self.output_device_index
        self.sample_rate = AudioCapture.choose_sample_rate(self._device_info()["device"])
        self.fft_size = fft_size
        self.audio_block_size = AUDIO_BLOCK_SIZE
        self.analyzer = SpectrumAnalyzer(self.sample_rate, fft_size)
        self._spectrum_queue: Queue[tuple[np.ndarray, np.ndarray]] = Queue(maxsize=2)
        self._analysis_queue: Queue[np.ndarray] = Queue(maxsize=2)
        self._state_lock = Lock()
        self._switch_lock = Lock()
        self._stop_event = Event()
        self._stream: sd.Stream | None = None
        self._input_stream: sd.InputStream | None = None
        self._left_stream: sd.OutputStream | None = None
        self._right_stream: sd.OutputStream | None = None
        self._left_queue: Queue[np.ndarray] = Queue(maxsize=8)
        self._right_queue: Queue[np.ndarray] = Queue(maxsize=8)
        self._active = False
        self._enabled = True
        self._mode = "Manual"
        self._gains = list(PRESETS["Flat"])
        self._latest_spectrum: tuple[np.ndarray, np.ndarray] | None = None
        self._last_error: str | None = None
        self._filter_state = np.zeros((len(BANDS), 2, 2), dtype=np.float32)
        self._smoothed_gains = np.zeros(len(BANDS), dtype=np.float32)
        self._volume = 1.0
        self._balance = 0.0
        self._lr_swap = False
        self._features = {"clarity": 0.0, "bass_boost": 0.0, "dynamic_boost": 0.0, "ambience": 0.0, "surround": 0.0}
        self._calibration = [1.0, 1.0]
        self._speaker_gains = [np.zeros(len(BANDS), dtype=np.float32), np.zeros(len(BANDS), dtype=np.float32)]
        self._speaker_filter_state = np.zeros((2, len(BANDS), 3), dtype=np.float32)

    def _preferred_device(self) -> int:
        return self.devices[0][0]

    def _device_info(self) -> dict:
        return dict(next(info for index, info in self.devices if index == self.device_index))
        

    def _preferred_output_device(self) -> int:
        default_name = ""
        try:
            wasapi_index = next(
                index
                for index, hostapi in enumerate(sd.query_hostapis())
                if hostapi["name"] == "Windows WASAPI"
            )
            default_id = sd.query_hostapis()[wasapi_index]["default_output_device"]
            default_name = sd.query_devices(default_id)["name"].casefold()
            for index, info in self.output_devices:
                if info["name"].casefold() == default_name:
                    return index
        except (RuntimeError, OSError, StopIteration):
            pass
        for index, info in self.output_devices:
            if info["name"].casefold() != default_name:
                return index
        return self.output_devices[0][0]

    def _output_device_info(self) -> dict:
        return dict(next(info for index, info in self.output_devices if index == self.output_device_index))

    def start(self) -> None:
        with self._switch_lock:
            if self._active:
                return
            self._stop_event.clear()
            self._last_error = None
            self._analysis_worker = Thread(target=self._analysis_loop, name="auto-eq-analysis", daemon=True)
            self._active = True
            self._analysis_worker.start()
            input_device = self._device_info()["device"]
            if self.left_output_device_index == self.right_output_device_index:
                output_device = self._output_device_info()["device"]
                self._stream = sd.Stream(
                    device=(input_device, output_device), samplerate=self.sample_rate,
                    blocksize=self.audio_block_size, channels=2, dtype="float32",
                    latency="high", callback=self._audio_callback,
                )
                self._stream.start()
            else:
                self._input_stream = sd.InputStream(
                    device=input_device, samplerate=self.sample_rate,
                    blocksize=self.audio_block_size, channels=2, dtype="float32",
                    latency="high", callback=self._dual_input_callback,
                )
                self._left_stream = sd.OutputStream(
                    device=self.left_output_device_index, samplerate=self.sample_rate,
                    blocksize=self.audio_block_size, channels=1, dtype="float32",
                    latency="high", callback=self._left_output_callback,
                )
                self._right_stream = sd.OutputStream(
                    device=self.right_output_device_index, samplerate=self.sample_rate,
                    blocksize=self.audio_block_size, channels=1, dtype="float32",
                    latency="high", callback=self._right_output_callback,
                )
                self._input_stream.start()
                self._left_stream.start()
                self._right_stream.start()

    def stop(self) -> None:
        with self._switch_lock:
            self._stop_event.set()
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            for stream_name in ("_input_stream", "_left_stream", "_right_stream"):
                stream = getattr(self, stream_name)
                if stream is not None:
                    stream.stop()
                    stream.close()
                    setattr(self, stream_name, None)
            if getattr(self, "_analysis_worker", None) is not None:
                self._analysis_worker.join(timeout=2.0)
                self._analysis_worker = None
            self._active = False

    def close(self) -> None:
        self.stop()

    def switch_device(self, device_index: int) -> None:
        if device_index == self.device_index:
            return
        was_active = self._active
        self.stop()
        self.device_index = device_index
        self.sample_rate = AudioCapture.choose_sample_rate(self._device_info()["device"])
        self.analyzer = SpectrumAnalyzer(self.sample_rate, self.fft_size)
        if was_active or self._last_error:
            self.start()

    def switch_output_device(self, device_index: int) -> None:
        if device_index == self.output_device_index:
            return
        was_active = self._active
        self.stop()
        self.output_device_index = device_index
        self.left_output_device_index = device_index
        self.right_output_device_index = device_index
        if was_active or self._last_error:
            self.start()

    def set_channel_outputs(self, left_device_index: int, right_device_index: int) -> None:
        was_active = self._active
        self.stop()
        self.left_output_device_index = left_device_index
        self.right_output_device_index = right_device_index
        self.output_device_index = right_device_index
        if was_active:
            self.start()

    def set_enabled(self, enabled: bool) -> None:
        with self._state_lock:
            self._enabled = enabled

    def set_mode(self, mode: str) -> None:
        if mode != "Manual":
            raise ValueError(f"Bilinmeyen EQ modu: {mode}")
        with self._state_lock:
            self._mode = mode

    def set_gain(self, band_index: int, value: float) -> None:
        with self._state_lock:
            self._gains[band_index] = float(np.clip(value, -12.0, 12.0))

    def apply_preset(self, name: str) -> None:
        if name not in PRESETS:
            return
        with self._state_lock:
            self._gains = list(PRESETS[name])

    def apply_genre(self, genre: str) -> None:
        if genre in GENRE_PROFILES:
            with self._state_lock:
                self._gains = list(GENRE_PROFILES[genre])

    def genre_profile_for(self, genre: str | None) -> str:
        normalized = (genre or "").casefold().strip()
        if not normalized:
            return "Flat"
        if "turkish pop" in normalized:
            return "Turkish Pop"
        if "hip hop" in normalized or "rap" in normalized or "trap" in normalized:
            return "Hip-Hop"
        if any(value in normalized for value in ("electronic", "edm", "house", "techno", "dubstep")):
            return "Electronic"
        if any(value in normalized for value in ("classical", "orchestra", "opera")):
            return "Classical"
        if any(value in normalized for value in ("rock", "metal", "punk")):
            return "Rock"
        if "podcast" in normalized or "spoken" in normalized:
            return "Podcast"
        if "pop" in normalized or "dance" in normalized:
            return "Pop"
        return "Flat"

    def apply_song_profile(self, song_id: str) -> None:
        self.apply_genre("Pop" if song_id else "Flat")

    def set_feature(self, name: str, value: float) -> None:
        if name in self._features:
            with self._state_lock:
                self._features[name] = float(np.clip(value, 0.0, 1.0))

    def set_volume(self, value: float) -> None:
        with self._state_lock:
            self._volume = float(np.clip(value, 0.0, 1.0))

    def set_balance(self, value: float) -> None:
        with self._state_lock:
            self._balance = float(np.clip(value, -1.0, 1.0))

    def set_lr_swap(self, enabled: bool) -> None:
        with self._state_lock:
            self._lr_swap = enabled

    def set_speaker_gain(self, channel: int, band_index: int, value: float) -> None:
        if channel not in (0, 1) or band_index not in range(len(BANDS)):
            return
        with self._state_lock:
            self._speaker_gains[channel][band_index] = float(np.clip(value, -12.0, 12.0))

    def set_speaker_calibration(self, left: float, right: float) -> None:
        with self._state_lock:
            self._calibration = [float(np.clip(left, 0.0, 2.0)), float(np.clip(right, 0.0, 2.0))]

    def snapshot(self) -> EngineSnapshot:
        with self._state_lock:
            gains = tuple(self._gains)
            enabled = self._enabled
            mode = self._mode
        info = self._device_info()
        output_info = self._output_device_info()
        return EngineSnapshot(
            active=self._active,
            enabled=enabled,
            mode=mode,
            gains=gains,
            spectrum=self._latest_spectrum,
            sample_rate=self.sample_rate,
            latency_ms=(self.audio_block_size / self.sample_rate) * 1000.0,
            input_name=info["name"],
            output_name=output_info["name"],
            error=self._last_error,
        )

    def _audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, timing: object, status: object) -> None:
        del frames, timing, status
        try:
            self._analysis_queue.put_nowait(indata.copy())
        except Full:
            try:
                self._analysis_queue.get_nowait()
                self._analysis_queue.put_nowait(indata.copy())
            except (Empty, Full):
                pass
        outdata[:] = self._process_audio(indata, outdata.shape[1])

    def _dual_input_callback(self, indata: np.ndarray, frames: int, timing: object, status: object) -> None:
        del frames, timing, status
        processed = self._process_audio(indata, 2)
        try:
            self._analysis_queue.put_nowait(indata.copy())
        except Full:
            pass
        for queue, channel in ((self._left_queue, processed[:, :1]), (self._right_queue, processed[:, 1:2])):
            try:
                queue.put_nowait(channel.copy())
            except Full:
                try:
                    queue.get_nowait()
                    queue.put_nowait(channel.copy())
                except (Empty, Full):
                    pass

    def _right_output_callback(self, outdata: np.ndarray, frames: int, timing: object, status: object) -> None:
        del frames, timing, status
        try:
            outdata[:] = self._right_queue.get_nowait()
        except Empty:
            outdata.fill(0)

    def _left_output_callback(self, outdata: np.ndarray, frames: int, timing: object, status: object) -> None:
        del frames, timing, status
        try:
            outdata[:] = self._left_queue.get_nowait()
        except Empty:
            outdata.fill(0)

    def _analysis_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                audio_block = self._analysis_queue.get(timeout=0.1)
            except Empty:
                continue
            frequencies, values = self.analyzer.analyze(audio_block)
            self._latest_spectrum = (frequencies, values)
            try:
                self._spectrum_queue.put_nowait((frequencies, values))
            except Full:
                try:
                    self._spectrum_queue.get_nowait()
                    self._spectrum_queue.put_nowait((frequencies, values))
                except (Empty, Full):
                    pass

    def _process_audio(self, audio_block: np.ndarray, output_channels: int) -> np.ndarray:
        with self._state_lock:
            enabled = self._enabled
            gains = np.asarray(self._gains, dtype=np.float32)
            speaker_gains = [values.copy() for values in self._speaker_gains]
            features = dict(self._features)
            volume = self._volume
            balance = self._balance
            lr_swap = self._lr_swap
            calibration = tuple(self._calibration)
        audio = np.asarray(audio_block, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        if audio.shape[1] == 1 and output_channels == 2:
            audio = np.repeat(audio, 2, axis=1)
        audio = audio[:, :output_channels].copy()
        if not enabled:
            return self._finish_output(audio, volume, balance, lr_swap, calibration)

        smoothing = 1.0 - np.exp(-len(audio) / (self.sample_rate * 0.5))
        gains = gains.copy()
        gains[0] += features["bass_boost"] * 6.0
        gains[2] += features["clarity"] * 2.0
        gains[3] += features["clarity"] * 3.0
        gains[4] += features["clarity"] * 1.5
        self._smoothed_gains += smoothing * (np.asarray(gains, dtype=np.float32) - self._smoothed_gains)
        centers = (60.0, 250.0, 1000.0, 4000.0, 12000.0)
        for band, (frequency, gain) in enumerate(zip(centers, self._smoothed_gains)):
            if abs(gain) < 0.05:
                continue
            b0, b1, b2, a1, a2 = self._biquad_coefficients(frequency, gain)
            for channel in range(audio.shape[1]):
                audio[:, channel], self._filter_state[band, channel] = lfilter(
                    (b0, b1, b2),
                    (1.0, a1, a2),
                    audio[:, channel],
                    zi=self._filter_state[band, channel],
                )
        speaker_smoothing = 1.0 - np.exp(-len(audio) / (self.sample_rate * 0.15))
        for channel in range(min(2, audio.shape[1])):
            for band, (frequency, target) in enumerate(zip(centers, speaker_gains[channel])):
                state = self._speaker_filter_state[channel, band]
                current = float(state[0])
                gain = current + speaker_smoothing * (float(target) - current)
                self._speaker_filter_state[channel, band, 0] = gain
                if abs(gain) < 0.05:
                    continue
                b0, b1, b2, a1, a2 = self._biquad_coefficients(frequency, gain)
                audio[:, channel], self._speaker_filter_state[channel, band, 1:] = lfilter(
                    (b0, b1, b2), (1.0, a1, a2), audio[:, channel], zi=self._speaker_filter_state[channel, band, 1:]
                )
        if features["dynamic_boost"] > 0.0:
            threshold = 0.75 - features["dynamic_boost"] * 0.25
            level = np.max(np.abs(audio), axis=1, keepdims=True)
            compression = np.minimum(1.0, threshold / np.maximum(level, threshold))
            audio *= 1.0 + (1.0 - compression) * features["dynamic_boost"] * 0.35
        if features["surround"] > 0.0 and audio.shape[1] >= 2:
            mid = (audio[:, 0] + audio[:, 1]) * 0.5
            side = (audio[:, 0] - audio[:, 1]) * 0.5
            side *= 1.0 + features["surround"] * 0.8
            audio[:, 0] = mid + side
            audio[:, 1] = mid - side
        if features["ambience"] > 0.0 and audio.shape[1] >= 2:
            delayed = np.vstack((np.zeros((1, 2), dtype=np.float32), audio[:-1]))
            audio += delayed * (features["ambience"] * 0.12)
        return self._finish_output(audio, volume, balance, lr_swap, calibration)

    @staticmethod
    def _finish_output(audio: np.ndarray, volume: float, balance: float, lr_swap: bool, calibration: tuple[float, float]) -> np.ndarray:
        if audio.shape[1] >= 2:
            audio[:, 0] *= calibration[0] * (1.0 - max(balance, 0.0))
            audio[:, 1] *= calibration[1] * (1.0 + min(balance, 0.0))
            if lr_swap:
                audio = audio[:, ::-1]
        return np.clip(audio * volume, -1.0, 1.0)

    def _biquad_coefficients(self, frequency: float, gain: float) -> tuple[float, ...]:
        w0 = 2.0 * np.pi * frequency / self.sample_rate
        alpha = np.sin(w0) / (2.0 * 0.7)
        amplitude = 10.0 ** (gain / 40.0)
        b0 = 1.0 + alpha * amplitude
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0 - alpha * amplitude
        a0 = 1.0 + alpha / amplitude
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha / amplitude
        return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0

    def _set_error(self, message: str) -> None:
        with self._state_lock:
            self._last_error = message
        self._active = False