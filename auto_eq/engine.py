"""Threaded audio facade shared by the desktop UI and future EQ DSP."""

from __future__ import annotations

import math

import json
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any

import numpy as np
import sounddevice as sd
from scipy.signal import lfilter


# ---------------------------------------------------------------------------
# Inline AudioCapture — WASAPI öncelikli VB-CABLE loopback ve çıkış cihaz keşfi
# ---------------------------------------------------------------------------
class AudioCapture:
    """Ses cihaz yardımcısı — Windows WASAPI öncelikli VB-CABLE ve çıkış cihazlarını listeler."""

    @staticmethod
    def _wasapi_hostapi_id() -> int | None:
        try:
            for idx, api in enumerate(sd.query_hostapis()):
                if "wasapi" in api.get("name", "").casefold():
                    return idx
        except Exception:
            pass
        return None

    @classmethod
    def list_devices(cls) -> list[tuple[int, dict]]:
        """Giriş (loopback) olarak kullanılabilecek cihazları döndür.
        Önce WASAPI VB-CABLE arar, bulamazsa diğer HostAPI CABLE veya giriş cihazlarını döndürür."""
        try:
            devices = sd.query_devices()
        except Exception:
            return []

        wasapi_id = cls._wasapi_hostapi_id()
        wasapi_cables: list[tuple[int, dict]] = []
        other_cables: list[tuple[int, dict]] = []
        wasapi_inputs: list[tuple[int, dict]] = []
        other_inputs: list[tuple[int, dict]] = []

        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                name = dev.get("name", "").casefold()
                is_wasapi = (wasapi_id is not None and dev.get("hostapi") == wasapi_id)
                info = (i, {"device": i, "name": dev["name"], "hostapi": dev.get("hostapi")})

                if "cable output" in name or ("cable" in name and "input" not in name):
                    if is_wasapi:
                        wasapi_cables.append(info)
                    else:
                        other_cables.append(info)
                elif "cable" in name:
                    if is_wasapi:
                        wasapi_cables.append(info)
                    else:
                        other_cables.append(info)
                else:
                    if is_wasapi:
                        wasapi_inputs.append(info)
                    else:
                        other_inputs.append(info)

        result = wasapi_cables + other_cables
        if not result:
            result = wasapi_inputs + other_inputs
        return result

    @classmethod
    def list_output_devices(cls) -> list[tuple[int, dict]]:
        """Çıkış olarak kullanılabilecek cihazları döndür (WASAPI öncelikli, Cable hariç)."""
        try:
            devices = sd.query_devices()
        except Exception:
            return []

        wasapi_id = cls._wasapi_hostapi_id()
        wasapi_outputs: list[tuple[int, dict]] = []
        other_outputs: list[tuple[int, dict]] = []

        for i, dev in enumerate(devices):
            if dev.get("max_output_channels", 0) > 0:
                name = dev.get("name", "").casefold()
                if "cable" in name:
                    continue
                info = (i, {"device": i, "name": dev["name"], "hostapi": dev.get("hostapi")})
                if wasapi_id is not None and dev.get("hostapi") == wasapi_id:
                    wasapi_outputs.append(info)
                else:
                    other_outputs.append(info)

        return wasapi_outputs if wasapi_outputs else other_outputs

    @staticmethod
    def choose_sample_rate(device_index: int) -> float:
        """Cihaz için desteklenen en iyi örnekleme hızını seç (48000 veya 44100)."""
        try:
            info = sd.query_devices(device_index)
            default_sr = int(info.get("default_samplerate", 48000))
            for sr in (48000, 44100, 96000, 22050):
                try:
                    sd.check_input_settings(device=device_index, samplerate=sr, channels=2)
                    return float(sr)
                except Exception:
                    pass
            return float(default_sr)
        except Exception:
            return 48000.0


# ---------------------------------------------------------------------------
# Inline SpectrumAnalyzer — FFT tabanlı frekans analizi
# ---------------------------------------------------------------------------
class SpectrumAnalyzer:
    """Hızlı rfft tabanlı spektrum analizörü — log-spaced 128 bin, dBFS çıktısı."""

    def __init__(self, sample_rate: float, fft_size: int = 2048) -> None:
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self._window = np.hanning(fft_size).astype(np.float32)
        # Çıkış frekans ekseni (DC bileşeni hariç)
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
        self._freqs = freqs[1:].astype(np.float32)  # DC'yi at
        self._ref = float(fft_size)  # Normalizasyon referansı
        self._history = np.zeros(fft_size, dtype=np.float32)

    def analyze(self, audio_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ses bloğundan (N, 2) veya (N,) frekans + dBFS değerleri üret."""
        data = np.asarray(audio_block, dtype=np.float32)
        if data.ndim == 2:
            data = data.mean(axis=1)  # Stereo → mono
        n = len(data)
        if n >= self.fft_size:
            self._history[:] = data[-self.fft_size:]
        elif n > 0:
            self._history = np.roll(self._history, -n)
            self._history[-n:] = data
        frame = self._history * self._window
        spectrum = np.fft.rfft(frame)
        magnitude = np.abs(spectrum[1:]) / self._ref  # DC'yi at
        # dBFS: küçük magnitude değerlerin log'unu önlemek için epsilon ekle
        db = 20.0 * np.log10(np.maximum(magnitude, 1e-9)).astype(np.float32)
        db = np.nan_to_num(db, nan=-90.0, posinf=0.0, neginf=-90.0)
        return self._freqs, db

BANDS = ("Bass", "Low Mid", "Mid", "High Mid", "Treble")
AUDIO_BLOCK_SIZE = 1024

# Sabit varsayılan profiller (kullanıcı override'ı yoksa bunlar kullanılır)
_DEFAULT_GENRE_PROFILES: dict[str, tuple[float, ...]] = {
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

_DEFAULT_GENRE_DSP: dict[str, dict[str, float]] = {
    "Flat": {"clarity": 0.0, "bass_boost": 0.0, "dynamic_boost": 0.0, "ambience": 0.0, "surround": 0.0},
    "Pop": {"clarity": 0.20, "bass_boost": 0.20, "dynamic_boost": 0.15, "ambience": 0.10, "surround": 0.15},
    "Turkish Pop": {"clarity": 0.25, "bass_boost": 0.25, "dynamic_boost": 0.15, "ambience": 0.10, "surround": 0.20},
    "Rock": {"clarity": 0.35, "bass_boost": 0.30, "dynamic_boost": 0.20, "ambience": 0.15, "surround": 0.25},
    "Metal": {"clarity": 0.40, "bass_boost": 0.35, "dynamic_boost": 0.25, "ambience": 0.10, "surround": 0.20},
    "Jazz": {"clarity": 0.30, "bass_boost": 0.15, "dynamic_boost": 0.05, "ambience": 0.25, "surround": 0.20},
    "Classical": {"clarity": 0.30, "bass_boost": 0.10, "dynamic_boost": 0.0, "ambience": 0.35, "surround": 0.30},
    "Hip-Hop": {"clarity": 0.20, "bass_boost": 0.60, "dynamic_boost": 0.25, "ambience": 0.05, "surround": 0.15},
    "Electronic": {"clarity": 0.35, "bass_boost": 0.50, "dynamic_boost": 0.25, "ambience": 0.15, "surround": 0.35},
    "Podcast": {"clarity": 0.50, "bass_boost": 0.0, "dynamic_boost": 0.20, "ambience": 0.0, "surround": 0.0},
}

_DEFAULT_PRESETS: dict[str, tuple[float, ...]] = {
    "Flat": (0.0, 0.0, 0.0, 0.0, 0.0),
    "Music": (2.0, 1.0, 0.0, 1.5, 2.5),
    "Bass Boost": (7.0, 3.0, 0.0, -1.0, 1.0),
    "Vocal": (-1.0, 1.0, 3.0, 2.0, 0.0),
    "Gaming": (3.0, 1.0, -1.0, 2.0, 3.0),
    "Movie": (4.0, 1.0, 0.0, 1.0, 4.0),
}

_DEFAULT_PRESET_DSP: dict[str, dict[str, float]] = {
    "Flat": {"clarity": 0.0, "bass_boost": 0.0, "dynamic_boost": 0.0, "ambience": 0.0, "surround": 0.0},
    "Music": {"clarity": 0.25, "bass_boost": 0.20, "dynamic_boost": 0.10, "ambience": 0.15, "surround": 0.20},
    "Bass Boost": {"clarity": 0.10, "bass_boost": 0.70, "dynamic_boost": 0.20, "ambience": 0.0, "surround": 0.10},
    "Vocal": {"clarity": 0.45, "bass_boost": 0.0, "dynamic_boost": 0.15, "ambience": 0.05, "surround": 0.05},
    "Gaming": {"clarity": 0.40, "bass_boost": 0.30, "dynamic_boost": 0.25, "ambience": 0.10, "surround": 0.40},
    "Movie": {"clarity": 0.30, "bass_boost": 0.40, "dynamic_boost": 0.20, "ambience": 0.30, "surround": 0.45},
}

_PROFILES_PATH = Path.home() / ".auralab_profiles.json"


def _load_profiles() -> tuple[
    dict[str, tuple[float, ...]],
    dict[str, tuple[float, ...]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, Any],
]:
    """Kaydedilmiş kullanıcı profillerini ve hoparlör ayarlarını dosyadan yükle."""
    genre_overrides: dict[str, tuple[float, ...]] = {}
    preset_overrides: dict[str, tuple[float, ...]] = {}
    genre_dsp_overrides: dict[str, dict[str, float]] = {}
    preset_dsp_overrides: dict[str, dict[str, float]] = {}
    speaker_settings: dict[str, Any] = {}

    if _PROFILES_PATH.exists():
        try:
            data = json.loads(_PROFILES_PATH.read_text(encoding="utf-8"))
            for key, values in data.get("genre_overrides", {}).items():
                genre_overrides[key] = tuple(float(v) for v in values)
            for key, values in data.get("preset_overrides", {}).items():
                preset_overrides[key] = tuple(float(v) for v in values)
            for key, dsp in data.get("genre_dsp_overrides", {}).items():
                genre_dsp_overrides[key] = {k: float(v) for k, v in dsp.items()}
            for key, dsp in data.get("preset_dsp_overrides", {}).items():
                preset_dsp_overrides[key] = {k: float(v) for k, v in dsp.items()}
            speaker_settings = data.get("speaker_settings", {})
        except Exception:
            pass
    return genre_overrides, preset_overrides, genre_dsp_overrides, preset_dsp_overrides, speaker_settings


def _save_profiles(
    genre_overrides: dict[str, tuple[float, ...]],
    preset_overrides: dict[str, tuple[float, ...]],
    genre_dsp_overrides: dict[str, dict[str, float]],
    preset_dsp_overrides: dict[str, dict[str, float]],
    speaker_settings: dict[str, Any],
) -> None:
    """Kullanıcı override'larını ve donanım ayarlarını dosyaya kaydet."""
    data = {
        "genre_overrides": {k: list(v) for k, v in genre_overrides.items()},
        "preset_overrides": {k: list(v) for k, v in preset_overrides.items()},
        "genre_dsp_overrides": genre_dsp_overrides,
        "preset_dsp_overrides": preset_dsp_overrides,
        "speaker_settings": speaker_settings,
    }
    _PROFILES_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _merged_genre_profiles(
    genre_overrides: dict[str, tuple[float, ...]],
) -> dict[str, tuple[float, ...]]:
    merged = dict(_DEFAULT_GENRE_PROFILES)
    merged.update(genre_overrides)
    return merged


def _merged_presets(
    preset_overrides: dict[str, tuple[float, ...]],
) -> dict[str, tuple[float, ...]]:
    merged = dict(_DEFAULT_PRESETS)
    merged.update(preset_overrides)
    return merged


# Modül düzeyinde erişim için — UI'nin import ettiği isimler
_genre_ov, _preset_ov, _genre_dsp_ov, _preset_dsp_ov, _spk_set = _load_profiles()
GENRE_PROFILES: dict[str, tuple[float, ...]] = _merged_genre_profiles(_genre_ov)
PRESETS: dict[str, tuple[float, ...]] = _merged_presets(_preset_ov)


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
        self._spectrum_queue: Queue[tuple[np.ndarray, np.ndarray]] = Queue(maxsize=4)
        self._analysis_queue: Queue[np.ndarray] = Queue(maxsize=4)
        self._state_lock = Lock()
        self._switch_lock = Lock()
        self._stop_event = Event()
        self._stream: sd.Stream | None = None
        self._input_stream: sd.InputStream | None = None
        self._left_stream: sd.OutputStream | None = None
        self._right_stream: sd.OutputStream | None = None
        self._left_queue: Queue[np.ndarray] = Queue(maxsize=16)
        self._right_queue: Queue[np.ndarray] = Queue(maxsize=16)
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
        # Kanal başına DSP özellikleri (sol=0, sağ=1)
        self._speaker_features: list[dict[str, float]] = [
            {"clarity": 0.0, "bass_boost": 0.0},
            {"clarity": 0.0, "bass_boost": 0.0},
        ]

        # Profil ve donanım ayarlarını yükle
        (
            self._genre_overrides,
            self._preset_overrides,
            self._genre_dsp_overrides,
            self._preset_dsp_overrides,
            self._speaker_settings,
        ) = _load_profiles()

        # Kaydedilmiş hoparlör kalibrasyonu ve ayarlarını geri yükle
        sp_cal = self._speaker_settings.get("calibration")
        if sp_cal and len(sp_cal) == 2:
            self._calibration = [float(sp_cal[0]), float(sp_cal[1])]

        sp_gains = self._speaker_settings.get("speaker_gains")
        if sp_gains and len(sp_gains) == 2 and len(sp_gains[0]) == len(BANDS):
            self._speaker_gains = [
                np.array(sp_gains[0], dtype=np.float32),
                np.array(sp_gains[1], dtype=np.float32),
            ]

        sp_feat = self._speaker_settings.get("speaker_features")
        if sp_feat and len(sp_feat) == 2:
            self._speaker_features = [
                {"clarity": float(sp_feat[0].get("clarity", 0.0)), "bass_boost": float(sp_feat[0].get("bass_boost", 0.0))},
                {"clarity": float(sp_feat[1].get("clarity", 0.0)), "bass_boost": float(sp_feat[1].get("bass_boost", 0.0))},
            ]

        if "balance" in self._speaker_settings:
            self._balance = float(self._speaker_settings["balance"])
        if "lr_swap" in self._speaker_settings:
            self._lr_swap = bool(self._speaker_settings["lr_swap"])

        # Kaydedilmiş çıkış aygıtları geçerliyse seç
        valid_indices = [idx for idx, _ in self.output_devices]
        saved_left = self._speaker_settings.get("left_output_device_index")
        saved_right = self._speaker_settings.get("right_output_device_index")
        is_separate = bool(self._speaker_settings.get("is_separate", False))
        if is_separate and saved_left in valid_indices and saved_right in valid_indices and saved_left != saved_right:
            self.left_output_device_index = saved_left
            self.right_output_device_index = saved_right
            self.output_device_index = saved_right
        else:
            # Varsayılan olarak R/L ayrık gelmesin, tek/birleşik stereo çıkış olsun
            self.left_output_device_index = self.output_device_index
            self.right_output_device_index = self.output_device_index
            self._speaker_settings["is_separate"] = False

    def _preferred_device(self) -> int:
        return self.devices[0][0]

    def _device_info(self) -> dict:
        return dict(next(info for index, info in self.devices if index == self.device_index))

    def _preferred_output_device(self) -> int:
        """C-Media cihazını öncelikle seç, yoksa WASAPI varsayılanına ya da ilk cihaza dön."""
        wasapi_id = AudioCapture._wasapi_hostapi_id()

        # 1. Önce WASAPI C-Media ara
        for index, info in self.output_devices:
            if wasapi_id is not None and info.get("hostapi") == wasapi_id:
                name = info["name"].casefold()
                if "c-media" in name or "cmedia" in name or "c media" in name:
                    return index

        # 2. Herhangi bir host API C-Media ara
        for index, info in self.output_devices:
            name = info["name"].casefold()
            if "c-media" in name or "cmedia" in name or "c media" in name:
                return index

        # 3. WASAPI varsayılan çıkışına bak
        default_name = ""
        try:
            if wasapi_id is not None:
                default_id = sd.query_hostapis()[wasapi_id]["default_output_device"]
                default_name = sd.query_devices(default_id)["name"].casefold()
                for index, info in self.output_devices:
                    if info["name"].casefold() == default_name:
                        return index
        except (RuntimeError, OSError, StopIteration, KeyError, IndexError):
            pass

        # 4. WASAPI fiziksel (Cable olmayan) ilk cihaz
        for index, info in self.output_devices:
            if wasapi_id is not None and info.get("hostapi") == wasapi_id:
                name = info["name"].casefold()
                if not any(v in name for v in ("cable", "virtual", "steam", "loopback", "voice")):
                    return index

        # 5. VB-CABLE olmayan ilk cihaz
        for index, info in self.output_devices:
            if "cable" not in info["name"].casefold():
                return index

        return self.output_devices[0][0]

    def _output_device_info(self) -> dict:
        return dict(next(info for index, info in self.output_devices if index == self.output_device_index))

    # ------------------------------------------------------------------
    # Profil ve Donanım Ayarlarını Kaydetme / Yükleme
    # ------------------------------------------------------------------

    def _persist_all(self) -> None:
        """Tüm override ve ayarları diske yaz."""
        _save_profiles(
            self._genre_overrides,
            self._preset_overrides,
            self._genre_dsp_overrides,
            self._preset_dsp_overrides,
            self._speaker_settings,
        )

    def save_genre_profile(self, name: str, gains: tuple[float, ...], dsp: dict[str, float] | None = None) -> None:
        """Mevcut EQ değerlerini ve DSP renklerini verilen genre profili olarak kaydet."""
        with self._state_lock:
            self._genre_overrides[name] = tuple(gains)
            if dsp is not None:
                self._genre_dsp_overrides[name] = dict(dsp)
        self._persist_all()
        GENRE_PROFILES[name] = tuple(gains)

    def save_preset(self, name: str, gains: tuple[float, ...], dsp: dict[str, float] | None = None) -> None:
        """Mevcut EQ değerlerini ve DSP ayarlarını verilen preset olarak kaydet."""
        with self._state_lock:
            self._preset_overrides[name] = tuple(gains)
            if dsp is not None:
                self._preset_dsp_overrides[name] = dict(dsp)
        self._persist_all()
        PRESETS[name] = tuple(gains)

    def save_speaker_settings(self) -> None:
        """Kullanıcının hoparlör kalibrasyonu, speaker EQ, DSP ve yönlendirme ayarlarını kaydet."""
        with self._state_lock:
            self._speaker_settings = {
                "calibration": list(self._calibration),
                "speaker_gains": [self._speaker_gains[0].tolist(), self._speaker_gains[1].tolist()],
                "speaker_features": [dict(self._speaker_features[0]), dict(self._speaker_features[1])],
                "balance": float(self._balance),
                "lr_swap": bool(self._lr_swap),
                "is_separate": bool(self.left_output_device_index != self.right_output_device_index),
                "left_output_device_index": self.left_output_device_index,
                "right_output_device_index": self.right_output_device_index,
            }
        self._persist_all()

    def effective_genre_gains(self, name: str) -> tuple[float, ...]:
        return self._genre_overrides.get(name, _DEFAULT_GENRE_PROFILES.get(name, (0.0,) * len(BANDS)))

    def effective_preset_gains(self, name: str) -> tuple[float, ...]:
        return self._preset_overrides.get(name, _DEFAULT_PRESETS.get(name, (0.0,) * len(BANDS)))

    def effective_preset_dsp(self, name: str) -> dict[str, float]:
        if name in self._preset_dsp_overrides:
            return dict(self._preset_dsp_overrides[name])
        return dict(_DEFAULT_PRESET_DSP.get(name, {
            "clarity": 0.0, "bass_boost": 0.0, "dynamic_boost": 0.0, "ambience": 0.0, "surround": 0.0
        }))

    def effective_genre_dsp(self, name: str) -> dict[str, float]:
        if name in self._genre_dsp_overrides:
            return dict(self._genre_dsp_overrides[name])
        return dict(_DEFAULT_GENRE_DSP.get(name, {
            "clarity": 0.0, "bass_boost": 0.0, "dynamic_boost": 0.0, "ambience": 0.0, "surround": 0.0
        }))

    def get_features(self) -> dict[str, float]:
        with self._state_lock:
            return dict(self._features)

    def get_speaker_settings(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "calibration": list(self._calibration),
                "speaker_gains": [self._speaker_gains[0].copy(), self._speaker_gains[1].copy()],
                "speaker_features": [dict(self._speaker_features[0]), dict(self._speaker_features[1])],
                "balance": self._balance,
                "lr_swap": self._lr_swap,
            }

    def has_genre_override(self, name: str) -> bool:
        return name in self._genre_overrides

    def has_preset_override(self, name: str) -> bool:
        return name in self._preset_overrides

    # ------------------------------------------------------------------
    # Engine control
    # ------------------------------------------------------------------

    def _cleanup_streams(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        for stream_name in ("_input_stream", "_left_stream", "_right_stream"):
            stream = getattr(self, stream_name, None)
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                setattr(self, stream_name, None)

    def start(self) -> None:
        with self._switch_lock:
            if self._active:
                return
            self._stop_event.clear()
            self._last_error = None
            input_device = self._device_info()["device"]

            # Kuyrukları sıfırla
            while not self._left_queue.empty():
                try:
                    self._left_queue.get_nowait()
                except (Empty, Exception):
                    break
            while not self._right_queue.empty():
                try:
                    self._right_queue.get_nowait()
                except (Empty, Exception):
                    break

            try:
                if self.left_output_device_index == self.right_output_device_index:
                    output_device = self._output_device_info()["device"]
                    self._stream = sd.Stream(
                        device=(input_device, output_device),
                        samplerate=self.sample_rate,
                        blocksize=self.audio_block_size,
                        channels=2,
                        dtype="float32",
                        latency="high",
                        callback=self._audio_callback,
                    )
                    self._stream.start()
                else:
                    # Pre-buffer silence in queues to prevent startup underflow pops
                    for _ in range(4):
                        self._left_queue.put_nowait(np.zeros((self.audio_block_size, 2), dtype=np.float32))
                        self._right_queue.put_nowait(np.zeros((self.audio_block_size, 2), dtype=np.float32))

                    self._input_stream = sd.InputStream(
                        device=input_device,
                        samplerate=self.sample_rate,
                        blocksize=self.audio_block_size,
                        channels=2,
                        dtype="float32",
                        latency="high",
                        callback=self._dual_input_callback,
                    )
                    self._left_stream = sd.OutputStream(
                        device=self.left_output_device_index,
                        samplerate=self.sample_rate,
                        blocksize=self.audio_block_size,
                        channels=2,
                        dtype="float32",
                        latency="high",
                        callback=self._left_output_callback,
                    )
                    self._right_stream = sd.OutputStream(
                        device=self.right_output_device_index,
                        samplerate=self.sample_rate,
                        blocksize=self.audio_block_size,
                        channels=2,
                        dtype="float32",
                        latency="high",
                        callback=self._right_output_callback,
                    )
                    self._input_stream.start()
                    self._left_stream.start()
                    self._right_stream.start()

                self._active = True
                self._analysis_worker = Thread(target=self._analysis_loop, name="auto-eq-analysis", daemon=True)
                self._analysis_worker.start()
            except Exception as err:
                self._cleanup_streams()
                self._active = False
                # Eğer ayrık modda hata oluştuysa ana birleşik stereo cihaza geri dön
                if self.left_output_device_index != self.right_output_device_index:
                    try:
                        self.left_output_device_index = self.output_device_index
                        self.right_output_device_index = self.output_device_index
                        self.save_speaker_settings()
                        output_device = self._output_device_info()["device"]
                        self._stream = sd.Stream(
                            device=(input_device, output_device),
                            samplerate=self.sample_rate,
                            blocksize=self.audio_block_size,
                            channels=2,
                            dtype="float32",
                            latency="high",
                            callback=self._audio_callback,
                        )
                        self._stream.start()
                        self._active = True
                        self._last_error = f"Ayrık çıkış başlatılamadı ({err}). Birleşik stereo moda geçildi."
                        self._analysis_worker = Thread(target=self._analysis_loop, name="auto-eq-analysis", daemon=True)
                        self._analysis_worker.start()
                        return
                    except Exception as fb_err:
                        self._last_error = f"Ses motoru başlatılamadı: {fb_err}"
                        self._active = False
                else:
                    self._last_error = f"Ses motoru başlatılamadı: {err}"

    def stop(self) -> None:
        with self._switch_lock:
            self._stop_event.set()
            self._cleanup_streams()
            if getattr(self, "_analysis_worker", None) is not None:
                try:
                    self._analysis_worker.join(timeout=1.0)
                except Exception:
                    pass
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
        if device_index == self.output_device_index and self.left_output_device_index == device_index and self.right_output_device_index == device_index:
            return
        was_active = self._active
        self.stop()
        self.output_device_index = device_index
        self.left_output_device_index = device_index
        self.right_output_device_index = device_index
        self.save_speaker_settings()
        if was_active or self._last_error:
            self.start()

    def set_channel_outputs(self, left_device_index: int, right_device_index: int) -> None:
        was_active = self._active
        self.stop()
        self.left_output_device_index = left_device_index
        self.right_output_device_index = right_device_index
        self.output_device_index = right_device_index
        self.save_speaker_settings()
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

    def apply_preset(self, name: str) -> dict[str, float]:
        """Preset kazançlarını uygula, o preset'in DSP özelliklerini motora aktar ve döndür."""
        gains = self.effective_preset_gains(name)
        if gains:
            with self._state_lock:
                self._gains = list(gains)
        dsp = self.effective_preset_dsp(name)
        with self._state_lock:
            for k, v in dsp.items():
                if k in self._features:
                    self._features[k] = float(np.clip(v, 0.0, 1.0))
            features_copy = dict(self._features)
        return features_copy

    def apply_genre(self, genre: str) -> dict[str, float]:
        """Genre kazançlarını uygula, o genre'nın DSP özelliklerini motora aktar ve döndür."""
        gains = self.effective_genre_gains(genre)
        if gains:
            with self._state_lock:
                self._gains = list(gains)
        dsp = self.effective_genre_dsp(genre)
        with self._state_lock:
            for k, v in dsp.items():
                if k in self._features:
                    self._features[k] = float(np.clip(v, 0.0, 1.0))
            features_copy = dict(self._features)
        return features_copy

    def register_or_get_genre(self, raw_genre: str | None) -> str:
        """Raw genre metnini normalize edip genre profiline eşler.
        Eğer eşleşen mevcut bir profil yoksa yeni profil adı oluşturur, Flat (0 dB) olarak kaydeder ve döner."""
        if not raw_genre or not raw_genre.strip():
            return "Flat"
        
        raw_clean = raw_genre.strip()
        # 1. Tam veya küçük harf eşleşme var mı kontrol et
        for existing in list(GENRE_PROFILES.keys()):
            if existing.casefold() == raw_clean.casefold():
                return existing
        
        # 2. Bilinen anahtar kelimelerden biriyle örtüşüyor mu?
        known = self._match_known_genre_keyword(raw_clean)
        if known and known in GENRE_PROFILES:
            return known
            
        # 3. Tamamen yeni bir tür! Adını düzgün formatla
        genre_name = " ".join(word.capitalize() for word in raw_clean.split())
        if genre_name not in GENRE_PROFILES:
            # Henüz ayar eklenmediği için Flat (0 dB) ve sıfır DSP olarak ekle
            self._genre_overrides[genre_name] = (0.0,) * len(BANDS)
            self._genre_dsp_overrides[genre_name] = {
                "clarity": 0.0, "bass_boost": 0.0, "dynamic_boost": 0.0, "ambience": 0.0, "surround": 0.0
            }
            GENRE_PROFILES[genre_name] = self._genre_overrides[genre_name]
            self._persist_all()
        return genre_name

    @staticmethod
    def _match_known_genre_keyword(raw_genre: str) -> str | None:
        normalized = raw_genre.casefold()
        if "turkish pop" in normalized:
            return "Turkish Pop"
        if "hip hop" in normalized or "hip-hop" in normalized or "rap" in normalized or "trap" in normalized:
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
        return None

    def genre_profile_for(self, genre: str | None) -> str:
        return self.register_or_get_genre(genre)

    def apply_song_profile(self, song_id: str) -> None:
        self.apply_genre("Pop" if song_id else "Flat")

    def set_feature(self, name: str, value: float) -> None:
        if name in self._features:
            with self._state_lock:
                self._features[name] = float(np.clip(value, 0.0, 1.0))

    def set_speaker_feature(self, channel: int, name: str, value: float) -> None:
        """Kanal başına DSP özelliği ayarla (channel=0 sol, channel=1 sağ)."""
        if channel not in (0, 1):
            return
        if name not in self._speaker_features[channel]:
            return
        with self._state_lock:
            self._speaker_features[channel][name] = float(np.clip(value, 0.0, 1.0))

    def set_volume(self, value: float) -> None:
        with self._state_lock:
            self._volume = float(np.clip(value, 0.0, 1.5))

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

    def get_spectrum_fast(self) -> tuple[tuple[Any, Any] | None, tuple[float, ...], bool]:
        """Minimal-overhead read for the spectrum timer — no lock, no device info."""
        with self._state_lock:
            gains = tuple(self._gains)
            enabled = self._enabled
        return self._latest_spectrum, gains, enabled

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

    # ------------------------------------------------------------------
    # Audio callbacks
    # ------------------------------------------------------------------

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
        try:
            outdata[:] = self._process_audio(indata, outdata.shape[1])
        except Exception:
            outdata[:] = indata[:, :outdata.shape[1]]

    def _dual_input_callback(self, indata: np.ndarray, frames: int, timing: object, status: object) -> None:
        del frames, timing, status
        try:
            processed = self._process_audio(indata, 2)
            try:
                self._analysis_queue.put_nowait(indata.copy())
            except Full:
                try:
                    self._analysis_queue.get_nowait()
                    self._analysis_queue.put_nowait(indata.copy())
                except (Empty, Full):
                    pass

            # processed[:, 0] -> Sol kanal sinyali
            # processed[:, 1] -> Sağ kanal sinyali
            # Her bir hoparlör cihazına kendi atanmış kanalını iki çıkış kanalına da gönderiyoruz.
            # Böylece cihaz mono, stereo ya da tek hattan dinlese bile ses asla kesilmez/sessiz kalmaz.
            left_chunk = np.column_stack((processed[:, 0], processed[:, 0]))
            right_chunk = np.column_stack((processed[:, 1], processed[:, 1]))

            for queue, chunk in ((self._left_queue, left_chunk), (self._right_queue, right_chunk)):
                try:
                    queue.put_nowait(chunk)
                except Full:
                    try:
                        queue.get_nowait()
                        queue.put_nowait(chunk)
                    except (Empty, Full):
                        pass
        except Exception:
            pass

    def _right_output_callback(self, outdata: np.ndarray, frames: int, timing: object, status: object) -> None:
        del frames, timing, status
        try:
            block = self._right_queue.get_nowait()
            if block.shape == outdata.shape:
                outdata[:] = block
            elif block.shape[0] == outdata.shape[0]:
                outdata[:] = block[:, :outdata.shape[1]]
            else:
                outdata.fill(0)
        except (Empty, ValueError, Exception):
            outdata.fill(0)

    def _left_output_callback(self, outdata: np.ndarray, frames: int, timing: object, status: object) -> None:
        del frames, timing, status
        try:
            block = self._left_queue.get_nowait()
            if block.shape == outdata.shape:
                outdata[:] = block
            elif block.shape[0] == outdata.shape[0]:
                outdata[:] = block[:, :outdata.shape[1]]
            else:
                outdata.fill(0)
        except (Empty, ValueError, Exception):
            outdata.fill(0)

    def _analysis_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                audio_block = self._analysis_queue.get(timeout=0.06)
            except Empty:
                # Ses durduğunda grafiğin havada takılı kalmasını önlemek için yumuşak sönümleme
                if self._latest_spectrum is not None:
                    freqs, vals = self._latest_spectrum
                    if np.any(vals > -88.0):
                        decayed = np.maximum(vals - 5.0, -90.0)
                        self._latest_spectrum = (freqs, decayed)
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
            speaker_features = [dict(sf) for sf in self._speaker_features]
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
            # Kanal başına speaker_features uygula (speaker_gains üzerine eklenir)
            ch_sg = speaker_gains[channel].copy()
            sf = speaker_features[channel]
            ch_sg[0] += sf["bass_boost"] * 6.0
            ch_sg[2] += sf["clarity"] * 2.0
            ch_sg[3] += sf["clarity"] * 3.0
            ch_sg[4] += sf["clarity"] * 1.5

            for band, (frequency, target) in enumerate(zip(centers, ch_sg)):
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
        audio = np.nan_to_num(audio, copy=False)
        return self._finish_output(audio, volume, balance, lr_swap, calibration)

    @staticmethod
    def _finish_output(audio: np.ndarray, volume: float, balance: float, lr_swap: bool, calibration: tuple[float, float]) -> np.ndarray:
        if audio.shape[1] >= 2:
            if lr_swap:
                audio = audio[:, ::-1]
            # Cosine pan-law: prevents full channel cutoff at extreme balance
            theta = (balance + 1.0) * 0.5 * (math.pi * 0.5)  # 0..π/2
            pan_l = math.cos(theta)
            pan_r = math.sin(theta)
            audio[:, 0] *= calibration[0] * pan_l
            audio[:, 1] *= calibration[1] * pan_r
        audio *= volume
        # Şeffaf tepe sınırlayıcı (peak limiter): sert kare dalga kırpması ve patlamaları önler
        peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        if peak > 0.98:
            audio *= (0.98 / peak)
        return np.nan_to_num(audio, copy=False)

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