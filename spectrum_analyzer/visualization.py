"""Matplotlib visualization for the live spectrum."""

from __future__ import annotations

from queue import Empty, Queue

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .audio import AudioCapture
from .fft import SpectrumAnalyzer


class SpectrumWindow:
    def __init__(self, capture: AudioCapture, analyzer: SpectrumAnalyzer, device_name: str) -> None:
        self.capture = capture
        self.analyzer = analyzer
        self.audio_queue: Queue[np.ndarray] = Queue(maxsize=2)
        self.latest_db = np.full(np.count_nonzero(analyzer.display_mask), -100.0)
        self.figure, self.axis = plt.subplots(figsize=(11, 6))
        self.line, = self.axis.semilogx(analyzer.frequencies[analyzer.display_mask], self.latest_db)
        self.axis.set_title(f"Gerçek Zamanlı Spectrum | {device_name}")
        self.axis.set_xlabel("Frekans (Hz)")
        self.axis.set_ylabel("Seviye (dBFS)")
        self.axis.set_xlim(20, 20000)
        self.axis.set_ylim(-100, 0)
        self.axis.grid(True, which="both", alpha=0.25)
        self.figure.tight_layout()
        self.animation = FuncAnimation(self.figure, self._update, interval=30, blit=True, cache_frame_data=False)

    def run(self) -> None:
        self.capture.callback = self._receive_audio
        self.capture.start()
        try:
            plt.show()
        finally:
            self.capture.stop()

    def _receive_audio(self, audio_block: np.ndarray) -> None:
        if self.audio_queue.full():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                pass
        self.audio_queue.put_nowait(audio_block)

    def _update(self, _frame: int):
        try:
            while True:
                audio_block = self.audio_queue.get_nowait()
        except Empty:
            audio_block = None

        if audio_block is not None:
            _, self.latest_db = self.analyzer.analyze(audio_block)
            self.line.set_ydata(self.latest_db)
        return (self.line,)