"""Windows default audio routing, volume synchronization, and recovery."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from pycaw.constants import ERole
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

warnings.filterwarnings(
    "ignore",
    message="COMError attempting to get property",
    module="pycaw.utils",
)


@dataclass(frozen=True)
class WindowsAudioState:
    default_device_id: str
    default_device_name: str
    volume: float
    muted: bool


class SystemAudioManager:
    """Own the Windows audio state for one Auto EQ session."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._state_path = Path.home() / ".auto_eq_audio_recovery.json"
        self._original: WindowsAudioState | None = None
        self._managed_ids: list[str] = []
        self._output_cache = None
        self._restore_recovery()

    @staticmethod
    def _default_device():
        return AudioUtilities.GetSpeakers()

    def _all_output_devices(self):
        if self._output_cache is None:
            self._output_cache = [
                device for device in AudioUtilities.GetAllDevices()
                if device.id and device.FriendlyName and device.id.startswith("{0.0.0.")
            ]
        return self._output_cache

    def _find_cable(self):
        for device in self._all_output_devices():
            if "cable input" in device.FriendlyName.casefold():
                return device
        raise RuntimeError("VB-CABLE CABLE Input output cihazı bulunamadı.")

    def _find_by_id(self, device_id: str):
        return next((device for device in self._all_output_devices() if device.id == device_id), None)

    @staticmethod
    def _endpoint_volume(device):
        return device.EndpointVolume

    def _read_state(self) -> WindowsAudioState:
        device = self._default_device()
        endpoint = self._endpoint_volume(device)
        return WindowsAudioState(
            default_device_id=device.id,
            default_device_name=device.FriendlyName,
            volume=float(endpoint.GetMasterVolumeLevelScalar()),
            muted=bool(endpoint.GetMute()),
        )

    def _write_recovery(self) -> None:
        if self._original is None:
            return
        self._state_path.write_text(
            json.dumps(self._original.__dict__, indent=2),
            encoding="utf-8",
        )

    def _restore_recovery(self) -> None:
        if not self._state_path.exists():
            return
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
            self.set_default_device(state["default_device_id"])
            device = self._find_by_id(state["default_device_id"])
            if device is not None:
                device.EndpointVolume.SetMasterVolumeLevelScalar(float(state["volume"]), None)
                device.EndpointVolume.SetMute(bool(state["muted"]), None)
            self._state_path.unlink(missing_ok=True)
        except Exception:
            pass

    def begin(self) -> WindowsAudioState:
        with self._lock:
            self._original = self._read_state()
            self._write_recovery()
            cable = self._find_cable()
            self.set_default_device(cable.id)
            self._managed_ids = [cable.id]
            self.set_master(self._original.volume, self._original.muted)
            return self._original

    def set_default_device(self, device_id: str) -> None:
        AudioUtilities.SetDefaultDevice(
            device_id,
            roles=[ERole.eConsole, ERole.eMultimedia, ERole.eCommunications],
        )

    def _managed_endpoints(self):
        endpoints = []
        for device_id in self._managed_ids:
            try:
                device = next(device for device in self._all_output_devices() if device.id == device_id)
                endpoints.append(self._endpoint_volume(device))
            except StopIteration:
                pass
        return endpoints

    def set_master(self, volume: float, muted: bool) -> None:
        value = max(0.0, min(1.0, float(volume)))
        for endpoint in self._managed_endpoints():
            endpoint.SetMasterVolumeLevelScalar(value, None)
            endpoint.SetMute(bool(muted), None)
        if self._managed_ids:
            try:
                current = self._default_device()
                endpoint = self._endpoint_volume(current)
                endpoint.SetMasterVolumeLevelScalar(value, None)
                endpoint.SetMute(bool(muted), None)
            except (OSError, RuntimeError):
                pass

    def current_master(self) -> tuple[float, bool]:
        device = self._default_device()
        endpoint = self._endpoint_volume(device)
        return float(endpoint.GetMasterVolumeLevelScalar()), bool(endpoint.GetMute())

    def sync_to_physical(self, device_id: str) -> None:
        if device_id not in self._managed_ids:
            self._managed_ids.append(device_id)
        volume, muted = self.current_master()
        try:
            device = next(device for device in self._all_output_devices() if device.id == device_id)
            endpoint = self._endpoint_volume(device)
            endpoint.SetMasterVolumeLevelScalar(volume, None)
            endpoint.SetMute(muted, None)
        except StopIteration:
            raise RuntimeError("Seçilen fiziksel output Windows endpoint listesinde yok.")

    def sync_to_physical_id(self, device_id: str) -> None:
        device = next((item for item in self._all_output_devices() if item.id == device_id), None)
        if device is None:
            raise RuntimeError("Seçilen fiziksel output Windows endpoint listesinde yok.")
        if device_id not in self._managed_ids:
            self._managed_ids.append(device_id)
        volume, muted = self.current_master()
        device.EndpointVolume.SetMasterVolumeLevelScalar(volume, None)
        device.EndpointVolume.SetMute(muted, None)

    def restore(self) -> None:
        with self._lock:
            if self._original is None:
                return
            original = self._original
            try:
                self.set_default_device(original.default_device_id)
                device = self._default_device()
                endpoint = self._endpoint_volume(device)
                endpoint.SetMasterVolumeLevelScalar(original.volume, None)
                endpoint.SetMute(original.muted, None)
                self._state_path.unlink(missing_ok=True)
                self._original = None
            except (OSError, RuntimeError):
                self._write_recovery()
                raise
