"""Windows audio device discovery and real-time capture."""

from __future__ import annotations

from collections.abc import Callable
from threading import Event, Thread

import numpy as np
import sounddevice as sd


class AudioCapture:
    """Capture float32 audio blocks from a selectable sounddevice input."""

    def __init__(
        self,
        device: object,
        sample_rate: float,
        channels: int,
        block_size: int,
        callback: Callable[[np.ndarray], None],
        error_callback: Callable[[str], None],
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.callback = callback
        self.error_callback = error_callback
        self.recorder: object | None = None
        self._stop_event = Event()
        self._thread: Thread | None = None

    @staticmethod
    def _wasapi_devices() -> list[tuple[int, dict]]:
        devices = sd.query_devices()
        wasapi_index = next(
            index
            for index, hostapi in enumerate(sd.query_hostapis())
            if hostapi["name"] == "Windows WASAPI"
        )
        return [
            (index, device)
            for index, device in enumerate(devices)
            if device["hostapi"] == wasapi_index
        ]

    @staticmethod
    def list_devices() -> list[tuple[int, dict]]:
        """Return the VB-CABLE recording endpoint used as the system-audio source."""
        devices = []
        index = 0
        for device_id, device in AudioCapture._wasapi_devices():
            if "cable output" not in device["name"].casefold() or device["max_input_channels"] <= 0:
                continue
            devices.append(
                (
                    index,
                    {
                        "name": f"VB-CABLE: {device['name']}",
                        "max_input_channels": device["max_input_channels"],
                        "max_output_channels": 0,
                        "default_samplerate": 48000.0,
                        "device": device_id,
                    },
                )
            )
            index += 1
        return devices

    @staticmethod
    def list_output_devices() -> list[tuple[int, dict]]:
        """Return physical Windows speaker endpoints for processed audio."""
        devices = []
        for device_id, device in AudioCapture._wasapi_devices():
            name = device["name"].casefold()
            if device["max_output_channels"] <= 0 or "cable" in name or "virtual cable" in name:
                continue
            devices.append(
                (
                    device_id,
                    {
                        "name": device["name"],
                        "max_input_channels": 0,
                        "max_output_channels": device["max_output_channels"],
                        "default_samplerate": 48000.0,
                        "device": device_id,
                    },
                )
            )
        return devices

    @staticmethod
    def input_devices() -> list[tuple[int, dict]]:
        return [
            (index, info)
            for index, info in AudioCapture.list_devices()
            if info["max_input_channels"] > 0
        ]

    @staticmethod
    def choose_sample_rate(device: object, requested: float = 48000.0) -> float:
        """Use the native WASAPI rate used by VB-CABLE and physical endpoints."""
        del device, requested
        return 48000.0

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _record_loop(self) -> None:
        try:
            with self.device.recorder(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.block_size,
            ) as recorder:
                self.recorder = recorder
                while not self._stop_event.is_set():
                    self.callback(np.asarray(recorder.record(numframes=self.block_size), dtype=np.float32))
        except (RuntimeError, OSError, ValueError) as error:
            self.error_callback(f"Ses akışı başlatılamadı: {error}")
        finally:
            self.recorder = None