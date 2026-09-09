"""Modern PyQt6 desktop interface for the threaded audio engine."""

from __future__ import annotations

import sys
import os
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QLockFile, Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QInputDialog,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from .engine import BANDS, GENRE_PROFILES, PRESETS, AudioEngine
from .spotify import SpotifyIntegration
from .system_audio import SystemAudioManager


class SpectrumCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(8, 2.8), facecolor="#171a20")
        super().__init__(self.figure)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_facecolor("#171a20")
        self.axis.set_xscale("log")
        self.axis.set_xlim(20, 20000)
        self.axis.set_ylim(-90, 0)
        self.axis.grid(True, which="both", color="#343a46", alpha=0.45, linewidth=0.6)
        self.axis.tick_params(colors="#87909f", labelsize=8)
        for spine in self.axis.spines.values():
            spine.set_color("#343a46")
        self.axis.set_xlabel("Frequency (Hz)", color="#87909f", fontsize=8)
        self.raw_line, = self.axis.plot([], [], color="#687281", linewidth=0.9, alpha=0.65)
        self.line, = self.axis.plot([], [], color="#d3f36b", linewidth=1.8)
        self.fill = None
        self.figure.subplots_adjust(left=0.055, right=0.99, top=0.96, bottom=0.22)

    def update_spectrum(self, spectrum: tuple[Any, Any] | None, gains: tuple[float, ...], enabled: bool) -> None:
        if spectrum is None:
            return
        frequencies, values = spectrum
        eq_curve = self._eq_curve(frequencies, gains) if enabled else np.zeros_like(values)
        display_values = np.clip(values + eq_curve, -90.0, 0.0)
        self.raw_line.set_data(frequencies, np.clip(values, -90.0, 0.0))
        self.line.set_data(frequencies, display_values)
        if self.fill is not None:
            self.fill.remove()
        self.fill = self.axis.fill_between(frequencies, display_values, -90, color="#d3f36b", alpha=0.08)
        self.draw_idle()

    @staticmethod
    def _eq_curve(frequencies: np.ndarray, gains: tuple[float, ...]) -> np.ndarray:
        centers = (60.0, 250.0, 1000.0, 4000.0, 12000.0)
        widths = (0.42, 0.38, 0.38, 0.38, 0.42)
        log_frequencies = np.log10(frequencies)
        curve = np.zeros_like(frequencies, dtype=float)
        for gain, center, width in zip(gains, centers, widths):
            curve += gain * np.exp(-((log_frequencies - np.log10(center)) / width) ** 2)
        return curve


class SettingsDialog(QDialog):
    def __init__(self, snapshot: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auto EQ Settings")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        title = QLabel("Audio engine settings")
        title.setObjectName("dialogTitle")
        layout.addWidget(title)
        for label, value in (
            ("Output device", snapshot.output_name),
            ("Sample rate", f"{snapshot.sample_rate:.0f} Hz"),
            ("FFT latency", f"{snapshot.latency_ms:.1f} ms"),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            value_label = QLabel(value)
            value_label.setObjectName("dialogValue")
            row.addWidget(value_label, 1, Qt.AlignmentFlag.AlignRight)
            layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class AutoEQWindow(QMainWindow):
    def __init__(self, engine: AudioEngine, audio_manager: SystemAudioManager) -> None:
        super().__init__()
        self.engine = engine
        self.audio_manager = audio_manager
        self._allow_exit = False
        self._last_system_volume: tuple[float, bool] | None = None
        self.sliders: list[QSlider] = []
        self.value_labels: list[QLabel] = []
        self._updating_ui = False
        self.spotify = SpotifyIntegration()
        self._last_spotify_track_id: str | None = None
        self._current_spotify_genre = ""
        self.setWindowTitle("Auto EQ")
        self.setMinimumSize(1040, 720)
        self.resize(1180, 820)
        self._build_ui()
        self._apply_theme()
        self._refresh_devices()
        self._setup_tray()
        self.engine.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(66)
        self.system_timer = QTimer(self)
        self.system_timer.timeout.connect(self._refresh_system_audio)
        self.system_timer.start(250)
        self.spotify_timer = QTimer(self)
        self.spotify_timer.timeout.connect(self._refresh_spotify)
        self.spotify_timer.start(4000)
        QTimer.singleShot(100, self._refresh_spotify)

    def _setup_tray(self) -> None:
        self.tray = QSystemTrayIcon(self)
        self.tray.setToolTip("Auralab Audio")
        menu = QMenu(self)
        show_action = menu.addAction("Open Auralab")
        show_action.triggered.connect(self._show_from_tray)
        menu.addSeparator()
        exit_action = menu.addAction("Exit and restore audio")
        exit_action.triggered.connect(self._exit_from_tray)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda reason: self._show_from_tray()
            if reason == QSystemTrayIcon.ActivationReason.DoubleClick else None
        )
        self.tray.show()

    def _show_from_tray(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def _exit_from_tray(self) -> None:
        self._allow_exit = True
        self.close()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(34, 28, 34, 22)
        layout.setSpacing(18)

        header = QHBoxLayout()
        title = QLabel("AURALAB")
        title.setObjectName("appTitle")
        header.addWidget(title)
        product = QLabel("REAL-TIME AUDIO CONTROL")
        product.setObjectName("headerMeta")
        header.addWidget(product, 0, Qt.AlignmentFlag.AlignVCenter)
        header.addStretch()
        self.status_dot = QLabel("●  Audio Engine Active")
        self.status_dot.setObjectName("statusActive")
        header.addWidget(self.status_dot)
        settings = QPushButton("Settings  /  Device")
        settings.setObjectName("secondaryButton")
        settings.clicked.connect(self._show_settings)
        header.addWidget(settings)
        layout.addLayout(header)

        hero = QFrame()
        hero.setObjectName("hero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(24, 18, 24, 18)
        copy = QVBoxLayout()
        eyebrow = QLabel("DSP CONSOLE  /  REAL-TIME AUDIO")
        eyebrow.setObjectName("eyebrow")
        copy.addWidget(eyebrow)
        name = QLabel("Audio control.")
        name.setObjectName("heroTitle")
        copy.addWidget(name)
        self.mode_label = QLabel("EQ and spatial controls are ready")
        self.mode_label.setObjectName("muted")
        copy.addWidget(self.mode_label)
        self.track_label = QLabel("Spotify  /  waiting for playback")
        self.track_label.setObjectName("trackLabel")
        copy.addWidget(self.track_label)
        hero_layout.addLayout(copy)
        hero_layout.addStretch()
        self.toggle = QPushButton("ON")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setObjectName("powerToggle")
        self.toggle.clicked.connect(self._toggle_engine)
        hero_layout.addWidget(self.toggle, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(hero)

        spectrum_card = self._card("REAL-TIME SPECTRUM")
        spectrum_layout = spectrum_card.layout()
        self.canvas = SpectrumCanvas()
        self.canvas.setMinimumHeight(290)
        spectrum_layout.addWidget(self.canvas)
        layout.addWidget(spectrum_card, 3)

        control_row = QHBoxLayout()
        eq_card = self._card("EQUALIZER")
        eq_card.setMinimumWidth(600)
        eq_card.layout().setContentsMargins(18, 6, 18, 10)
        eq_grid = QGridLayout()
        eq_grid.setHorizontalSpacing(16)
        eq_grid.setVerticalSpacing(12)
        for index, band in enumerate(BANDS):
            label = QLabel(band)
            label.setObjectName("bandLabel")
            slider = QSlider(Qt.Orientation.Vertical)
            slider.setRange(-120, 120)
            slider.setValue(0)
            slider.setFixedHeight(190)
            value = QLabel("0.0 dB")
            value.setObjectName("gainValue")
            slider.valueChanged.connect(lambda value, i=index: self._gain_changed(i, value / 10.0))
            self.sliders.append(slider)
            self.value_labels.append(value)
            eq_grid.addWidget(label, 0, index, Qt.AlignmentFlag.AlignCenter)
            eq_grid.addWidget(slider, 1, index, Qt.AlignmentFlag.AlignCenter)
            eq_grid.addWidget(value, 2, index, Qt.AlignmentFlag.AlignCenter)
        eq_card.layout().addLayout(eq_grid)
        control_row.addWidget(eq_card, 3)

        mode_card = self._card("GENRE / DSP")
        mode_card.setMinimumWidth(360)
        mode_layout = mode_card.layout()
        genre_label = QLabel("GENRE EQ")
        genre_label.setObjectName("eyebrow")
        mode_layout.addWidget(genre_label)
        self.genre_combo = QComboBox()
        self.genre_combo.addItems(GENRE_PROFILES.keys())
        self.genre_combo.currentTextChanged.connect(self._genre_profile_changed)
        mode_layout.addWidget(self.genre_combo)
        self.detected_genre_label = QLabel("Detected Spotify genre: -")
        self.detected_genre_label.setObjectName("muted")
        mode_layout.addWidget(self.detected_genre_label)
        preset_label = QLabel("PRESETS")
        preset_label.setObjectName("eyebrow")
        mode_layout.addWidget(preset_label)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS.keys())
        self.preset_combo.currentTextChanged.connect(self._preset_changed)
        mode_layout.addWidget(self.preset_combo)
        feature_label = QLabel("DSP FEATURES")
        feature_label.setObjectName("eyebrow")
        mode_layout.addWidget(feature_label)
        self.feature_sliders: dict[str, QSlider] = {}
        for key, title in (("clarity", "Clarity"), ("bass_boost", "Bass Boost"), ("dynamic_boost", "Dynamic Boost"), ("ambience", "Ambience"), ("surround", "Surround")):
            row = QHBoxLayout()
            row.addWidget(QLabel(title))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.valueChanged.connect(lambda value, name=key: self.engine.set_feature(name, value / 100.0))
            row.addWidget(slider)
            mode_layout.addLayout(row)
            self.feature_sliders[key] = slider
        control_row.addWidget(mode_card, 1)
        layout.addLayout(control_row, 2)

        routing_card = self._card("STEREO ROUTING")
        routing = QHBoxLayout()
        routing.addWidget(QLabel("Left speaker"))
        self.left_output_combo = QComboBox()
        self.left_output_combo.activated.connect(lambda: self._stereo_routing_changed(True))
        routing.addWidget(self.left_output_combo, 1)
        routing.addWidget(QLabel("Right speaker"))
        self.right_output_combo = QComboBox()
        self.right_output_combo.activated.connect(lambda: self._stereo_routing_changed(True))
        routing.addWidget(self.right_output_combo, 1)
        self.stereo_switch = QPushButton("L/R separate")
        self.stereo_switch.setCheckable(True)
        self.stereo_switch.toggled.connect(self._stereo_routing_changed)
        routing.addWidget(self.stereo_switch)
        self.swap_button = QPushButton("Swap L/R")
        self.swap_button.setCheckable(True)
        self.swap_button.toggled.connect(self._swap_speakers)
        routing.addWidget(self.swap_button)
        routing_card.layout().addLayout(routing)
        speaker_eq = QGridLayout()
        speaker_eq.addWidget(QLabel("Speaker EQ"), 0, 0)
        speaker_eq.addWidget(QLabel("Left"), 0, 1)
        speaker_eq.addWidget(QLabel("Right"), 0, 2)
        self.speaker_sliders: dict[tuple[int, int], QSlider] = {}
        for band_index, band in enumerate(BANDS):
            speaker_eq.addWidget(QLabel(band), band_index + 1, 0)
            for channel in (0, 1):
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(-120, 120)
                slider.setValue(0)
                slider.valueChanged.connect(
                    lambda value, c=channel, b=band_index: self.engine.set_speaker_gain(c, b, value / 10.0)
                )
                speaker_eq.addWidget(slider, band_index + 1, channel + 1)
                self.speaker_sliders[(channel, band_index)] = slider
        routing_card.layout().addLayout(speaker_eq)
        layout.addWidget(routing_card)

        footer_card = QFrame()
        footer_card.setObjectName("routeBar")
        footer = QHBoxLayout(footer_card)
        footer.setContentsMargins(16, 10, 16, 10)
        footer.addWidget(QLabel("Volume"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 150)
        self.volume_slider.setValue(100)
        self.volume_slider.valueChanged.connect(self._volume_changed)
        footer.addWidget(self.volume_slider, 1)
        self.mute_button = QPushButton("Mute")
        self.mute_button.setCheckable(True)
        self.mute_button.toggled.connect(self._mute_changed)
        footer.addWidget(self.mute_button)
        footer.addWidget(QLabel("Balance"))
        self.balance_slider = QSlider(Qt.Orientation.Horizontal)
        self.balance_slider.setRange(-100, 100)
        self.balance_slider.setValue(0)
        self.balance_slider.valueChanged.connect(lambda value: self.engine.set_balance(value / 100.0))
        footer.addWidget(self.balance_slider, 1)
        footer.addStretch()
        self.tech_label = QLabel("48 kHz  •  42.7 ms latency")
        self.tech_label.setObjectName("muted")
        footer.addWidget(self.tech_label)
        layout.addWidget(footer_card)

    def _card(self, title: str) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 16)
        label = QLabel(title)
        label.setObjectName("sectionTitle")
        layout.addWidget(label)
        return card

    def _apply_theme(self) -> None:
        self.setStyleSheet("""
            QWidget { color: #e8edf2; font-family: 'Segoe UI'; font-size: 13px; }
            #root { background: #0d1014; }
            #appTitle { font-size: 27px; font-weight: 800; letter-spacing: 2px; color: #f4f7fa; }
            #headerMeta { color: #667482; font-size: 10px; font-weight: 700; letter-spacing: 1.4px; margin-left: 12px; }
            #statusActive { color: #b7ef63; font-weight: 700; background: #1b2a20; border: 1px solid #315236; border-radius: 12px; padding: 6px 10px; }
            #statusInactive { color: #84909c; font-weight: 700; background: #1a1f25; border: 1px solid #303943; border-radius: 12px; padding: 6px 10px; }
            #hero, #card { background: #151a20; border: 1px solid #26303a; border-radius: 10px; }
            #hero { background: #18221f; border-color: #2e4a38; }
            #routeBar { background: #12171c; border: 1px solid #26303a; border-radius: 8px; }
            #eyebrow, #sectionTitle { color: #778695; font-size: 10px; font-weight: 800; letter-spacing: 1.4px; }
            #heroTitle { color: #f5f8f4; font-size: 34px; font-weight: 800; }
            #muted { color: #8b98a5; }
            #trackLabel { color: #b7ef63; font-size: 12px; font-weight: 700; padding-top: 6px; }
            #powerToggle { background: #b7ef63; color: #10150e; border: 0; border-radius: 24px; min-width: 92px; min-height: 52px; font-size: 15px; font-weight: 800; }
            #powerToggle:checked { background: #b7ef63; }
            QPushButton { background: #202832; color: #e7edf2; border: 1px solid #34404b; border-radius: 6px; padding: 8px 12px; }
            QPushButton:hover { background: #29343f; border-color: #9fd35a; }
            #secondaryButton { background: #171d24; color: #b8c2cc; }
            #bandLabel { color: #aeb9c3; font-weight: 600; }
            #gainValue { color: #b7ef63; font-weight: 800; }
            QSlider::groove:vertical { background: #303b46; width: 5px; border-radius: 2px; }
            QSlider::handle:vertical { background: #b7ef63; height: 16px; margin: 0 -6px; border-radius: 8px; }
            QSlider::sub-page:vertical { background: #668d42; }
            QSlider::groove:horizontal { background: #303b46; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #b7ef63; width: 13px; margin: -5px 0; border-radius: 7px; }
            QComboBox { background: #202832; border: 1px solid #34404b; border-radius: 6px; padding: 7px 10px; min-width: 150px; }
            QComboBox:hover { border-color: #9fd35a; }
            #dialogTitle { font-size: 20px; font-weight: 800; }
            #dialogValue { color: #b7ef63; }
            QToolTip { background: #202832; color: #e8edf2; border: 1px solid #4b5966; }
        """)

    def _refresh_devices(self) -> None:
        for combo, selected in (
            (self.left_output_combo, self.engine.left_output_device_index),
            (self.right_output_combo, self.engine.right_output_device_index),
        ):
            combo.blockSignals(True)
            combo.clear()
            for index, info in self.engine.output_devices:
                combo.addItem(info["name"], index)
            row = next((row for row, (index, _) in enumerate(self.engine.output_devices) if index == selected), 0)
            combo.setCurrentIndex(row)
            combo.blockSignals(False)
            try:
                self.audio_manager.sync_to_physical(self.engine.output_devices[row][1]["name"])
            except RuntimeError:
                pass

    def _refresh(self) -> None:
        snapshot = self.engine.snapshot()
        self.canvas.update_spectrum(snapshot.spectrum, snapshot.gains, snapshot.enabled)
        self._updating_ui = True
        for slider, label, gain in zip(self.sliders, self.value_labels, snapshot.gains):
            slider.setValue(round(gain * 10))
            label.setText(f"{gain:+.1f} dB")
        self._updating_ui = False
        self.tech_label.setText(f"{snapshot.sample_rate:.0f} Hz  •  {snapshot.latency_ms:.1f} ms latency")
        if snapshot.active:
            self.status_dot.setObjectName("statusActive")
            self.status_dot.setText("●  Audio Engine Active")
        else:
            self.status_dot.setObjectName("statusInactive")
            self.status_dot.setText("●  Audio Engine Inactive")
        self.status_dot.style().unpolish(self.status_dot)
        self.status_dot.style().polish(self.status_dot)
        if snapshot.error:
            self.mode_label.setText(snapshot.error)

    def _refresh_system_audio(self) -> None:
        try:
            volume, muted = self.audio_manager.current_master()
            current = (round(volume, 3), muted)
            if current != self._last_system_volume:
                self._last_system_volume = current
                self.volume_slider.blockSignals(True)
                self.volume_slider.setValue(round(volume * 100))
                self.volume_slider.blockSignals(False)
                self.mute_button.blockSignals(True)
                self.mute_button.setChecked(muted)
                self.mute_button.blockSignals(False)
                self.engine.set_volume(0.0 if muted else volume)
                self.audio_manager.set_master(volume, muted)
        except (OSError, RuntimeError):
            pass

    def _volume_changed(self, value: int) -> None:
        volume = max(0.0, min(1.0, value / 100.0))
        try:
            self.audio_manager.set_master(volume, self.mute_button.isChecked())
            self.engine.set_volume(volume)
            self._last_system_volume = (round(volume, 3), self.mute_button.isChecked())
        except (OSError, RuntimeError) as error:
            self.mode_label.setText(f"Volume sync failed: {error}")

    def _mute_changed(self, muted: bool) -> None:
        try:
            volume, _ = self.audio_manager.current_master()
            self.audio_manager.set_master(volume, muted)
            self._last_system_volume = (round(volume, 3), muted)
        except (OSError, RuntimeError) as error:
            self.mode_label.setText(f"Mute sync failed: {error}")

    def _refresh_spotify(self) -> None:
        track = self.spotify.current_track()
        if track is None or track.track_id == self._last_spotify_track_id:
            return
        self._last_spotify_track_id = track.track_id
        genre = self.engine.genre_profile_for(track.genre)
        self.engine.apply_genre(genre)
        genre_text = track.genre or "unknown genre"
        self.track_label.setText(f"Spotify  /  {track.title}  -  {track.artist}  /  {genre_text}")
        self.detected_genre_label.setText(f"Detected Spotify genre: {genre_text}")
        self.genre_combo.setCurrentText(genre)
        self.mode_label.setText(f"Genre profile applied: {genre}")

    def _gain_changed(self, index: int, value: float) -> None:
        self.value_labels[index].setText(f"{value:+.1f} dB")
        if not self._updating_ui:
            self.engine.set_gain(index, value)

    def _preset_changed(self, name: str) -> None:
        self.engine.apply_preset(name)

    def _genre_profile_changed(self, profile: str) -> None:
        self.engine.apply_genre(profile)

    def _toggle_engine(self, checked: bool) -> None:
        self.engine.set_enabled(checked)
        self.toggle.setText("ON" if checked else "OFF")
        self.mode_label.setText("DSP processing active" if checked else "Processing bypassed")

    def _output_changed(self, row: int) -> None:
        del row

    def _stereo_routing_changed(self, separate: bool) -> None:
        self.stereo_switch.setText("L/R separate" if separate else "Single output")
        if not separate:
            left = self.left_output_combo.currentData()
            if left is not None:
                ThreadTarget(lambda: self.engine.switch_output_device(int(left)))
            return
        left = self.left_output_combo.currentData()
        right = self.right_output_combo.currentData()
        if left is not None and right is not None:
            try:
                left_name = next(info["name"] for index, info in self.engine.output_devices if index == left)
                right_name = next(info["name"] for index, info in self.engine.output_devices if index == right)
                self.audio_manager.sync_to_physical(left_name)
                self.audio_manager.sync_to_physical(right_name)
            except (OSError, RuntimeError) as error:
                self.mode_label.setText(f"Stereo routing failed: {error}")
            ThreadTarget(lambda: self.engine.set_channel_outputs(int(left), int(right)))

    def _swap_speakers(self, enabled: bool) -> None:
        left = self.left_output_combo.currentData()
        right = self.right_output_combo.currentData()
        if left is None or right is None:
            return
        self.left_output_combo.blockSignals(True)
        self.right_output_combo.blockSignals(True)
        self.left_output_combo.setCurrentIndex(self.right_output_combo.currentIndex())
        self.right_output_combo.setCurrentIndex(next(
            (row for row, (index, _) in enumerate(self.engine.output_devices) if index == left), 0
        ))
        self.left_output_combo.blockSignals(False)
        self.right_output_combo.blockSignals(False)
        self.engine.set_lr_swap(False)
        self._stereo_routing_changed(True)

    def _show_settings(self) -> None:
        SettingsDialog(self.engine.snapshot(), self).exec()

    def closeEvent(self, event: Any) -> None:
        if not self._allow_exit:
            event.ignore()
            self.hide()
            return
        self.timer.stop()
        self.system_timer.stop()
        self.spotify_timer.stop()
        self.engine.close()
        self.tray.hide()
        try:
            self.audio_manager.restore()
        except Exception as error:
            QMessageBox.critical(self, "Audio restore failed", str(error))
        event.accept()


class ThreadTarget:
    def __init__(self, target: Any) -> None:
        import threading

        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Auto EQ")
    lock_path = os.path.join(os.environ.get("TEMP", "."), "auto_eq.lock")
    instance_lock = QLockFile(lock_path)
    instance_lock.setStaleLockTime(0)
    if not instance_lock.tryLock(0):
        QMessageBox.warning(None, "Auto EQ", "Auto EQ zaten çalışıyor.")
        return 1
    try:
        audio_manager = SystemAudioManager()
        audio_manager.begin()
        engine = AudioEngine()
        window = AutoEQWindow(engine, audio_manager)
        window.show()
        return app.exec()
    except Exception as error:
        try:
            if "audio_manager" in locals():
                audio_manager.restore()
        except Exception:
            pass
        QMessageBox.critical(None, "Auto EQ", f"Uygulama başlatılamadı:\n{error}")
        return 1
    finally:
        instance_lock.unlock()