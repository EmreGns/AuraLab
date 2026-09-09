"""FFT processing for spectrum visualization and future Auto EQ input."""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import hann


class SpectrumAnalyzer:
    def __init__(self, sample_rate: float, fft_size: int = 2048) -> None:
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.window = hann(fft_size, sym=False).astype(np.float32)
        self.frequencies = rfftfreq(fft_size, 1.0 / sample_rate)
        self.display_mask = (self.frequencies >= 20.0) & (self.frequencies <= 20000.0)

    def analyze(self, audio_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return frequency and dB arrays for the latest block."""
        if audio_block.ndim == 2:
            mono = np.mean(audio_block, axis=1)
        else:
            mono = audio_block

        if mono.size < self.fft_size:
            mono = np.pad(mono, (self.fft_size - mono.size, 0))
        frame = mono[-self.fft_size :].astype(np.float32, copy=False)
        spectrum = np.abs(rfft(frame * self.window))
        amplitude = (2.0 / np.sum(self.window)) * spectrum
        db = 20.0 * np.log10(np.maximum(amplitude, 1e-10))
        return self.frequencies[self.display_mask], db[self.display_mask]