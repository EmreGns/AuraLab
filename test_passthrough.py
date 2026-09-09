"""Direct CABLE loopback to a physical speaker, without Qt or EQ."""

from __future__ import annotations

import argparse
import time
import threading

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 48000
BLOCK_SIZE = 1024


def wasapi_devices() -> list[tuple[int, dict]]:
    devices = sd.query_devices()
    wasapi_index = next(
        index for index, hostapi in enumerate(sd.query_hostapis())
        if hostapi["name"] == "Windows WASAPI"
    )
    return [
        (index, device)
        for index, device in enumerate(devices)
        if device["hostapi"] == wasapi_index
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="EQ'suz doğrudan ses geçiş testi")
    parser.add_argument("--output", type=int, help="Çıkış cihazı numarası")
    parser.add_argument("--seconds", type=float, default=30.0, help="Test süresi")
    args = parser.parse_args()

    devices = wasapi_devices()
    source = next(
        (index for index, device in devices if "cable output" in device["name"].casefold()),
        None,
    )
    if source is None:
        raise RuntimeError("WASAPI CABLE Output kayıt cihazı bulunamadı.")

    outputs = [
        (index, device)
        for index, device in devices
        if device["max_output_channels"] > 0
        and "cable" not in device["name"].casefold()
    ]
    if not outputs:
        raise RuntimeError("Fiziksel çıkış cihazı bulunamadı.")

    print("EQ ve Qt kullanılmıyor. Doğrudan ses geçişi testi.")
    print(f"Kaynak: {sd.query_devices(source)['name']}")
    print("Çıkış cihazları:")
    for index, (_, device) in enumerate(outputs):
        print(f"  {index}: {device['name']}")

    output_index = args.output
    if output_index is None:
        output_index = 0
    if output_index < 0 or output_index >= len(outputs):
        raise ValueError("Geçersiz çıkış numarası.")

    output_device, output_info = outputs[output_index]
    print(f"Seçilen çıkış: {output_info['name']}")
    print(f"{args.seconds:.0f} saniye test başladı. Durdurmak için Ctrl+C.")

    callback_errors: list[str] = []
    blocks = 0
    lock = threading.Lock()

    def callback(indata: np.ndarray, outdata: np.ndarray, frames: int, timing: object, status: object) -> None:
        nonlocal blocks
        if status:
            with lock:
                callback_errors.append(str(status))
        outdata[:] = indata
        blocks += 1

    with sd.Stream(
        device=(source, output_device),
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=2,
        dtype="float32",
        latency="high",
        callback=callback,
    ):
        time.sleep(args.seconds)

    print(f"Tamamlandı: {blocks} blok, PortAudio uyarısı: {len(callback_errors)}")
    for error in sorted(set(callback_errors)):
        print(f"  {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
