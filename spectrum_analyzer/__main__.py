"""Command-line entry point for the spectrum analyzer."""

from __future__ import annotations

import argparse
import sys

from .audio import AudioCapture
from .fft import SpectrumAnalyzer
from .visualization import SpectrumWindow


def print_devices() -> list[tuple[int, dict]]:
    devices = AudioCapture.list_devices()
    print("Kullanılabilir ses cihazları:")
    print("ID   Giriş   Çıkış   Sample rate   Adı")
    for index, info in devices:
        print(
            f"{index:2d}   {info['max_input_channels']:>5}   "
            f"{info['max_output_channels']:>5}   "
            f"{info['default_samplerate']:>11.0f}   {info['name']}"
        )
    print("\nSistem sesi için 'Loopback:' ile başlayan cihazı seçin.")
    return devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gerçek zamanlı audio spectrum analyzer")
    parser.add_argument("--list-devices", action="store_true", help="Ses cihazlarını listele ve çık")
    parser.add_argument("--device", type=int, help="Kullanılacak ses cihazı ID'si")
    parser.add_argument("--sample-rate", type=float, default=44100.0)
    parser.add_argument("--fft-size", type=int, default=2048)
    return parser.parse_args()


def select_device(requested: int | None, devices: list[tuple[int, dict]]) -> tuple[int, dict]:
    input_devices = [(index, info) for index, info in devices if info["max_input_channels"] > 0]
    if not input_devices:
        raise RuntimeError("Input cihazı bulunamadı. Windows ses ayarlarında loopback/input cihazını etkinleştirin.")
    if requested is not None:
        selected = next(((index, info) for index, info in input_devices if index == requested), None)
        if selected is None:
            raise ValueError(f"Cihaz {requested} input olarak kullanılamıyor.")
        return selected
    if len(input_devices) == 1:
        return input_devices[0]
    choices = ", ".join(str(index) for index, _ in input_devices)
    while True:
        try:
            selected_id = int(input(f"Cihaz ID'si ({choices}): "))
            selected = next(((index, info) for index, info in input_devices if index == selected_id), None)
            if selected is not None:
                return selected
        except ValueError:
            pass
        print("Geçersiz input cihazı ID'si.")


def main() -> int:
    args = parse_args()
    try:
        devices = print_devices()
        if args.list_devices:
            return 0
        device_id, device_info = select_device(args.device, devices)
        sample_rate = AudioCapture.choose_sample_rate(device_info["device"], args.sample_rate)
        channels = min(2, int(device_info["max_input_channels"]))
        analyzer = SpectrumAnalyzer(sample_rate, args.fft_size)
        capture = AudioCapture(
            device_info["device"],
            sample_rate,
            channels,
            args.fft_size,
            callback=lambda _data: None,
            error_callback=lambda message: print(message, file=sys.stderr),
        )
        print(f"Başlatılıyor: {device_info['name']} | {sample_rate:.0f} Hz | {channels} kanal | FFT {args.fft_size}")
        SpectrumWindow(capture, analyzer, device_info["name"]).run()
        return 0
    except KeyboardInterrupt:
        print("\nDurduruldu.")
        return 0
    except (RuntimeError, ValueError, OSError) as error:
        print(f"HATA: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())