"""Windows default audio routing, volume synchronization, and recovery."""

from __future__ import annotations

import atexit
import json
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import comtypes
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
        self._restored = False
        atexit.register(self.restore)
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

    def find_preferred_physical_speaker(self):
        """Öncelikle C-Media cihazını, yoksa ilk gerçek fiziksel hoparlörü döndür."""
        devices = self._all_output_devices()
        # 1. C-Media öncelikli
        for device in devices:
            name = device.FriendlyName.casefold()
            if "c-media" in name or "cmedia" in name or "c media" in name:
                return device

        # 2. Virtual/Cable/Steam olmayan ilk fiziksel hoparlör
        for device in devices:
            name = device.FriendlyName.casefold()
            if not any(v in name for v in ("cable", "virtual", "steam", "loopback", "voice")):
                return device

        # 3. Cable olmayan herhangi biri
        for device in devices:
            name = device.FriendlyName.casefold()
            if "cable" not in name:
                return device

        return None

    def _find_by_id(self, device_id: str):
        return next((device for device in self._all_output_devices() if device.id == device_id), None)

    @staticmethod
    def _endpoint_volume(device):
        return device.EndpointVolume

    def _read_state(self) -> WindowsAudioState:
        device = self._default_device()
        # Eğer varsayılan aygıt şu anda zaten Cable veya sanal bir aygıtsa,
        # asla Cable'ı orijinal olarak kaydetme! Fiziksel hoparlörü bul.
        name_lower = device.FriendlyName.casefold()
        if "cable" in name_lower or "virtual" in name_lower:
            phys = self.find_preferred_physical_speaker()
            if phys is not None:
                device = phys

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
            dev_id = state.get("default_device_id")
            name = state.get("default_device_name", "").casefold()
            # Eğer kurtarma dosyasında yanlışlıkla cable varsa düzelt
            if "cable" in name or "virtual" in name:
                phys = self.find_preferred_physical_speaker()
                if phys is not None:
                    dev_id = phys.id
            if dev_id:
                self.set_default_device(dev_id)
                device = self._find_by_id(dev_id)
                if device is not None:
                    vol = float(state.get("volume", 0.5))
                    muted = bool(state.get("muted", False))
                    device.EndpointVolume.SetMasterVolumeLevelScalar(vol, None)
                    device.EndpointVolume.SetMute(muted, None)
            self._state_path.unlink(missing_ok=True)
        except Exception:
            pass

    def begin(self) -> WindowsAudioState:
        with self._lock:
            self._restored = False
            self._original = self._read_state()
            self._write_recovery()
            cable = self._find_cable()
            self.set_default_device(cable.id)
            self._managed_ids = [cable.id]
            # Kullanıcının mevcut Windows ses seviyesini anında uygula (geçiş sesi saklamasın)
            self.set_master(self._original.volume, self._original.muted)

        # Fiziksel hoparlörün donanım kanalını %100'e aç — 1 saniyelik gecikmeyle
        # (kullanıcı Windows ses kaydırıcısının anlık sıçramasını görmemelidir)
        threading.Thread(
            target=self._maximize_delayed,
            name="auralab-maximize-vol",
            daemon=True,
        ).start()

        return self._original

    def _maximize_delayed(self) -> None:
        """Arka planda 1 saniye bekleyip fiziksel cihaz sesini %100'e çek."""
        time.sleep(1.0)
        try:
            comtypes.CoInitialize()
        except Exception:
            pass
        try:
            self.maximize_physical_device_volume()
        except Exception:
            pass
        finally:
            try:
                comtypes.CoUninitialize()
            except Exception:
                pass

    def set_default_device(self, device_id: str) -> None:
        try:
            comtypes.CoInitialize()
        except Exception:
            pass
        try:
            AudioUtilities.SetDefaultDevice(
                device_id,
                roles=[ERole.eConsole, ERole.eMultimedia, ERole.eCommunications],
            )
        except Exception:
            pass

    def _managed_endpoints(self):
        endpoints = []
        for device_id in self._managed_ids:
            try:
                device = next(device for device in self._all_output_devices() if device.id == device_id)
                endpoints.append(self._endpoint_volume(device))
            except Exception:
                pass
        return endpoints

    def set_master(self, volume: float, muted: bool) -> None:
        value = max(0.0, min(1.0, float(volume)))
        for endpoint in self._managed_endpoints():
            try:
                endpoint.SetMasterVolumeLevelScalar(value, None)
                endpoint.SetMute(bool(muted), None)
            except Exception:
                pass
        if self._managed_ids:
            try:
                current = self._default_device()
                endpoint = self._endpoint_volume(current)
                endpoint.SetMasterVolumeLevelScalar(value, None)
                endpoint.SetMute(bool(muted), None)
            except Exception:
                pass

    def current_master(self) -> tuple[float, bool]:
        try:
            device = self._default_device()
            endpoint = self._endpoint_volume(device)
            return float(endpoint.GetMasterVolumeLevelScalar()), bool(endpoint.GetMute())
        except Exception:
            return 0.5, False

    def sync_to_physical(self, device_id: str) -> None:
        if device_id not in self._managed_ids:
            self._managed_ids.append(device_id)
        try:
            volume, muted = self.current_master()
            device = next((device for device in self._all_output_devices() if device.id == device_id or device.FriendlyName == device_id), None)
            if device is not None:
                endpoint = self._endpoint_volume(device)
                endpoint.SetMasterVolumeLevelScalar(volume, None)
                endpoint.SetMute(muted, None)
        except Exception:
            pass

    def maximize_physical_device_volume(self, device_name_or_id: str | None = None) -> None:
        """Fiziksel çıkış cihazlarının (örn. 24G4HRE, C-Media) Windows donanım sesini %100'e (1.0)
        alarak hoparlörlerin tam güçlerini kullanmasını sağlar."""
        try:
            for device in self._all_output_devices():
                try:
                    name = getattr(device, "FriendlyName", "") or ""
                    dev_id = getattr(device, "id", "") or ""
                    if any(v in name.casefold() for v in ("cable", "virtual", "steam", "line")):
                        continue
                    if device_name_or_id:
                        query = device_name_or_id.casefold().strip()
                        if query not in name.casefold() and query != dev_id:
                            continue
                    ep = self._endpoint_volume(device)
                    ep.SetMasterVolumeLevelScalar(1.0, None)
                    ep.SetMute(False, None)
                except Exception:
                    pass
        except Exception:
            pass

    def sync_to_physical_id(self, device_id: str) -> None:
        try:
            device = next((item for item in self._all_output_devices() if item.id == device_id), None)
            if device is None:
                return
            if device_id not in self._managed_ids:
                self._managed_ids.append(device_id)
            volume, muted = self.current_master()
            device.EndpointVolume.SetMasterVolumeLevelScalar(volume, None)
            device.EndpointVolume.SetMute(muted, None)
        except Exception:
            pass

    def restore(self) -> None:
        with self._lock:
            if self._restored:
                return
            target_id = None
            vol = 0.5
            muted = False
            if self._original is not None:
                orig_name = self._original.default_device_name.casefold()
                if "cable" not in orig_name and "virtual" not in orig_name:
                    target_id = self._original.default_device_id
                    vol = self._original.volume
                    muted = self._original.muted

            # Eğer original yoksa veya Cable ise, fiziksel hoparlöre dön (C-Media vb.)
            if target_id is None:
                phys = self.find_preferred_physical_speaker()
                if phys is not None:
                    target_id = phys.id

            if target_id:
                try:
                    self.set_default_device(target_id)
                    device = self._find_by_id(target_id)
                    if device is not None:
                        endpoint = self._endpoint_volume(device)
                        endpoint.SetMasterVolumeLevelScalar(vol, None)
                        endpoint.SetMute(muted, None)
                    self._state_path.unlink(missing_ok=True)
                    self._original = None
                    self._restored = True
                except (OSError, RuntimeError):
                    self._write_recovery()
                    raise

    def is_cable_default(self) -> bool:
        """Varsayılan çıkışın VB-CABLE olup olmadığını kontrol et."""
        try:
            device = self._default_device()
            return "cable input" in device.FriendlyName.casefold()
        except Exception:
            return False

    def switch_to_cable(self) -> str:
        """Windows varsayılan ses çıkışını VB-CABLE Input yaparak AutoEQ'yu aktif et."""
        cable = self._find_cable()
        self.set_default_device(cable.id)
        if cable.id not in self._managed_ids:
            self._managed_ids.append(cable.id)
        self._restored = False
        return cable.FriendlyName

    def switch_to_speaker(self) -> str:
        """Windows varsayılan ses çıkışını doğrudan fiziksel hoparlöre (C-Media) aktar."""
        phys = self.find_preferred_physical_speaker()
        if phys is None:
            raise RuntimeError("Fiziksel hoparlör bulunamadı.")
        self.set_default_device(phys.id)
        return phys.FriendlyName

    def toggle_routing(self) -> tuple[bool, str]:
        """Cable ile fiziksel hoparlör arasında geçiş yap. (is_cable, device_name) döndürür."""
        with self._lock:
            if self.is_cable_default():
                name = self.switch_to_speaker()
                return False, name
            else:
                name = self.switch_to_cable()
                return True, name

