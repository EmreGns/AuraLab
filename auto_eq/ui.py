"""Premium PyQt6 desktop interface — AuraLab v2.3."""

from __future__ import annotations

import atexit
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from PyQt6.QtCore import QLockFile, QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QIcon, QLinearGradient, QPainter, QPainterPath, QPalette, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QSystemTrayIcon,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .engine import BANDS, GENRE_PROFILES, PRESETS, AudioEngine
from .spotify import SpotifyIntegration
from .system_audio import SystemAudioManager
from .auralab_icon import create_auralab_icon


def _setup_crash_logger() -> None:
    crash_file = Path.cwd() / "crash.log"

    def _handle_exception(exc_type, exc_val, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_val, exc_tb)
            return
        tb_str = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
        sys.stderr.write(f"\n[AuraLab CRASH DETECTED]\n{tb_str}\n")
        sys.stderr.flush()
        try:
            with open(crash_file, "a", encoding="utf-8") as f:
                f.write(f"\n--- CRASH AT {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                f.write(tb_str)
        except Exception:
            pass

    sys.excepthook = _handle_exception
    try:
        import threading
        threading.excepthook = lambda args: _handle_exception(args.exc_type, args.exc_value, args.exc_traceback)
    except AttributeError:
        pass


_setup_crash_logger()

# ──────────────────────────────────────────────────────────────────────────────
# THEME SYSTEM
# ──────────────────────────────────────────────────────────────────────────────
BG        = "#070b0f"
BG2       = "#0c1318"
BG3       = "#111d28"
BORDER    = "#1a2a3a"
BORDER_HV = "#2a4a6a"
BLUE      = "#4a9eff"
BLUE_BG   = "#081828"
TEXT      = "#dce8f4"
TEXT2     = "#7a9ab8"
TEXT3     = "#3d5570"
WARN      = "#f5a623"

THEMES = {
    "Yeşil":     {"ACCENT": "#8df230", "ACCENT_DIM": "#4a8a18", "ACCENT_BG": "#0e1f0a"},
    "Mavi":      {"ACCENT": "#4a9eff", "ACCENT_DIM": "#2a6ab8", "ACCENT_BG": "#081828"},
    "Mor":       {"ACCENT": "#b47aff", "ACCENT_DIM": "#7a4ab8", "ACCENT_BG": "#180e28"},
    "Turuncu":   {"ACCENT": "#ff9a3e", "ACCENT_DIM": "#b86a18", "ACCENT_BG": "#281808"},
    "Kırmızı":   {"ACCENT": "#ff5a6a", "ACCENT_DIM": "#b83040", "ACCENT_BG": "#280e10"},
    "Camgöbeği": {"ACCENT": "#3edcdc", "ACCENT_DIM": "#1a9a9a", "ACCENT_BG": "#082020"},
}

_THEME_PATH = Path.home() / ".auralab_theme.json"

# Current accent colors (mutable at runtime)
ACCENT    = "#8df230"
ACCENT_DIM = "#4a8a18"
ACCENT_BG = "#0e1f0a"


def _load_theme() -> str:
    """Load saved theme name, or return default."""
    try:
        if _THEME_PATH.exists():
            data = json.loads(_THEME_PATH.read_text(encoding="utf-8"))
            name = data.get("theme", "Yeşil")
            if name in THEMES:
                return name
    except Exception:
        pass
    return "Yeşil"


def _save_theme(name: str) -> None:
    """Persist selected theme name."""
    try:
        _THEME_PATH.write_text(json.dumps({"theme": name}), encoding="utf-8")
    except Exception:
        pass



def _apply_accent(name: str) -> None:
    """Update global accent variables from theme name."""
    global ACCENT, ACCENT_DIM, ACCENT_BG
    t = THEMES.get(name, THEMES["Yeşil"])
    ACCENT = t["ACCENT"]
    ACCENT_DIM = t["ACCENT_DIM"]
    ACCENT_BG = t["ACCENT_BG"]


# Apply saved theme on import
_apply_accent(_load_theme())


# ──────────────────────────────────────────────────────────────────────────────
# SPECTRUM CANVAS
# ──────────────────────────────────────────────────────────────────────────────
class SpectrumCanvas(QWidget):
    """Akıcı 30 FPS QPainter spektrum görselleştirici — 96 bin, sahte-smooth animasyon, ön-hesaplanmış koordinatlar."""

    # Sabit padding değerleri
    _PAD_L = 32; _PAD_R = 38; _PAD_T = 12; _PAD_B = 22
    _NUM_BINS = 96

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(130)
        self.setMaximumHeight(210)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        # Frekans eksenini log-aralıklı ön-hesapla
        self._target_freqs = np.logspace(np.log10(20.0), np.log10(20000.0), self._NUM_BINS)
        self._x_norm = np.linspace(0.0, 1.0, self._NUM_BINS, dtype=np.float32)

        # Animasyon durumu
        self._display_vals = np.full(self._NUM_BINS, -90.0, dtype=np.float32)  # ekranda gösterilen
        self._target_vals  = np.full(self._NUM_BINS, -90.0, dtype=np.float32)  # FFT'den gelen hedef
        self._eq_curve_vals = np.zeros(self._NUM_BINS, dtype=np.float32)
        self._prev_gains: tuple[float, ...] = ()

        # Boyuta bağlı önbellek (resizeEvent'te güncellenir)
        self._cached_size: tuple[int, int] = (0, 0)
        self._cached_x_px: np.ndarray = np.zeros(self._NUM_BINS, dtype=np.float64)  # piksel X pozisyonları
        self._cached_plot_w: int = 0
        self._cached_plot_h: int = 0
        self._cached_bottom_y: float = 0.0

        # Log grid sabitleri
        self._log_min = float(np.log10(20.0))
        self._log_range = float(np.log10(20000.0) - self._log_min)

        # Çalışma-zamanı tampon dizileri (heap allocation önlemek için)
        self._tmp_ys = np.zeros(self._NUM_BINS, dtype=np.float64)
        self._tmp_eq_ys = np.zeros(self._NUM_BINS, dtype=np.float64)

    # ------------------------------------------------------------------
    def resizeEvent(self, event: Any) -> None:
        self._rebuild_cache()
        super().resizeEvent(event)

    def _rebuild_cache(self) -> None:
        w, h = self.width(), self.height()
        if (w, h) == self._cached_size or w <= 0 or h <= 0:
            return
        self._cached_size = (w, h)
        plot_w = max(10, w - self._PAD_L - self._PAD_R)
        plot_h = max(10, h - self._PAD_T - self._PAD_B)
        self._cached_plot_w = plot_w
        self._cached_plot_h = plot_h
        self._cached_bottom_y = float(self._PAD_T + plot_h)
        # Tüm X piksel pozisyonlarını tek seferde hesapla (artık her frame'de hesaplanmaz)
        self._cached_x_px = (self._PAD_L + self._x_norm * plot_w).astype(np.float64)

    # ------------------------------------------------------------------
    def update_spectrum(self, spectrum: tuple[Any, Any] | None, gains: tuple[float, ...], enabled: bool) -> None:
        """EQ eğrisi ve hedef spektrum değerlerini güncelle; animasyonu bir adım ilerlet."""
        # EQ eğrisi — yalnızca kazanç değiştiğinde yeniden hesapla
        if gains != self._prev_gains:
            self._prev_gains = gains
            if enabled:
                self._eq_curve_vals[:] = self._eq_curve(self._target_freqs, gains)
            else:
                self._eq_curve_vals.fill(0.0)

        # Hedef değerleri FFT verisinden güncelle
        if spectrum is not None:
            raw_freqs, raw_dbs = spectrum
            if len(raw_freqs) > 0 and len(raw_dbs) > 0:
                raw_dbs = np.nan_to_num(raw_dbs, nan=-90.0, posinf=0.0, neginf=-90.0)
                interp = np.interp(self._target_freqs, raw_freqs, raw_dbs,
                                   left=-90.0, right=-90.0).astype(np.float32)
                np.clip(interp + self._eq_curve_vals, -90.0, 0.0, out=self._target_vals)
            else:
                np.maximum(self._target_vals - 2.0, -90.0, out=self._target_vals)
        else:
            np.maximum(self._target_vals - 2.5, -90.0, out=self._target_vals)

        self._target_vals = np.nan_to_num(self._target_vals, nan=-90.0, posinf=0.0, neginf=-90.0)
        # Sahte-smooth animasyon: display_vals'ı target_vals'a doğru expotansiyel hareketle kaydır
        # Yükseliş hızlı (0.55), düşüş yavaş (0.14) → doğal bar hareketi
        diff = self._target_vals - self._display_vals
        # In-place işlem — heap allocation yok
        rise_mask = diff > 0
        self._display_vals[rise_mask]  += diff[rise_mask]  * 0.55
        self._display_vals[~rise_mask] += diff[~rise_mask] * 0.14
        self._display_vals = np.nan_to_num(self._display_vals, nan=-90.0, posinf=0.0, neginf=-90.0)

    def refresh_accent(self) -> None:
        self.update()

    # ------------------------------------------------------------------
    def paintEvent(self, event: Any) -> None:
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        # Önbellek boyut kontrolü (resizeEvent gelmediyse)
        if (w, h) != self._cached_size:
            self._rebuild_cache()

        pad_l  = self._PAD_L; pad_r  = self._PAD_R
        pad_t  = self._PAD_T; pad_b  = self._PAD_B
        plot_w = self._cached_plot_w
        plot_h = self._cached_plot_h
        bottom_y = self._cached_bottom_y
        x_px = self._cached_x_px  # ön-hesaplanmış X koordinatları

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Arka plan
        p.fillRect(0, 0, w, h, QColor(BG2))
        p.setPen(QPen(QColor(BORDER), 1.0))
        p.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), 8, 8)

        # dB yatay grid
        p.setFont(QFont("Consolas", 7))
        for db in (-20.0, -50.0, -80.0):
            y = pad_t + (1.0 - (db + 90.0) / 90.0) * plot_h
            p.setPen(QPen(QColor(BORDER), 1.0, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(pad_l, y), QPointF(pad_l + plot_w, y))
            p.setPen(QColor(TEXT3))
            p.drawText(QRectF(pad_l + plot_w + 4, y - 6, 32, 12),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"{int(db)}")

        # Frekans dikey grid
        for freq, label in ((60, "60"), (250, "250"), (1000, "1k"), (4000, "4k"), (12000, "12k")):
            xn = (math.log10(freq) - self._log_min) / self._log_range
            x  = pad_l + xn * plot_w
            p.setPen(QPen(QColor(BORDER), 1.0, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(x, pad_t), QPointF(x, bottom_y))
            p.setPen(QColor(TEXT3))
            p.drawText(QRectF(x - 15, bottom_y + 3, 30, 13), Qt.AlignmentFlag.AlignCenter, label)

        # EQ eğri kılavuz çizgisi (kesikli) — önbellek dizisine yaz
        if np.any(np.abs(self._eq_curve_vals) > 0.1):
            eq_dbs = np.clip(self._eq_curve_vals, -12.0, 12.0, out=self._tmp_eq_ys)
            np.multiply(1.0 - (eq_dbs + 45.0) / 90.0, plot_h, out=self._tmp_eq_ys)
            eq_ys = pad_t + self._tmp_eq_ys
            eq_path = QPainterPath()
            eq_path.moveTo(x_px[0], eq_ys[0])
            for i in range(1, self._NUM_BINS):
                eq_path.lineTo(x_px[i], eq_ys[i])
            p.setPen(QPen(QColor(ACCENT_DIM).lighter(140), 1.0, Qt.PenStyle.DashLine))
            p.drawPath(eq_path)

        # Dalga eğrisi — ön-hesaplanmış tampon dizisine yaz
        ys = self._tmp_ys
        np.multiply(1.0 - (self._display_vals.astype(np.float64) + 90.0) / 90.0, plot_h, out=ys)
        ys += pad_t  # in-place shift

        accent_color = QColor(ACCENT)
        x0 = x_px[0]; x_last = x_px[-1]

        line_path = QPainterPath()
        line_path.moveTo(x0, ys[0])
        for i in range(1, self._NUM_BINS):
            line_path.lineTo(x_px[i], ys[i])

        wave_path = QPainterPath(line_path)
        wave_path.lineTo(x_last, bottom_y)
        wave_path.lineTo(x0, bottom_y)
        wave_path.closeSubpath()

        # Gradyan dolgu
        grad = QLinearGradient(0.0, float(pad_t), 0.0, bottom_y)
        tc = QColor(accent_color); tc.setAlpha(115)
        mc = QColor(accent_color); mc.setAlpha(38)
        bc = QColor(accent_color); bc.setAlpha(4)
        grad.setColorAt(0.0, tc); grad.setColorAt(0.5, mc); grad.setColorAt(1.0, bc)
        p.fillPath(wave_path, grad)

        # Glow halo
        gc = QColor(accent_color); gc.setAlpha(55)
        p.setPen(QPen(gc, 3.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(line_path)

        # Keskin neon çizgi
        p.setPen(QPen(accent_color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(line_path)

    # ------------------------------------------------------------------
    @staticmethod
    def _eq_curve(freq: np.ndarray, gains: tuple[float, ...]) -> np.ndarray:
        log_centers = (1.77815125, 2.39794001, 3.0, 3.60205999, 4.07918125)  # 60, 250, 1k, 4k, 12k log10
        widths      = (0.42,       0.38,       0.38, 0.38,       0.42)
        lf  = np.log10(freq)
        out = np.zeros(len(freq), dtype=np.float32)
        for g, lc, w in zip(gains, log_centers, widths):
            if abs(g) > 1e-4:
                comp = g * np.exp(-((lf - lc) / w) ** 2)
                out += comp.astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# SETTINGS DIALOG
class SettingsDialog(QDialog):
    def __init__(self, engine: AudioEngine, audio_manager: SystemAudioManager,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.audio_manager = audio_manager
        self.setWindowTitle("AuraLab — Gelişmiş Hoparlör Ayarları")
        self.setMinimumSize(700, 560)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self._apply_dialog_theme()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Tab bar
        self.tabs = QTabWidget()
        self.tabs.setObjectName("settingsTabs")
        self.tabs.addTab(self._build_device_tab(), "⚙  Cihaz Bilgisi & Yönlendirme")
        self.tabs.addTab(self._build_speaker_eq_tab(), "📊  Speaker EQ (Kanal Başına)")
        self.tabs.addTab(self._build_speaker_dsp_tab(), "✨  Speaker DSP & Kalibrasyon")
        root.addWidget(self.tabs)

    # ── Device tab ────────────────────────────────────────────────────────────
    def _build_device_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(14)

        # Engine info
        snap = self.engine.snapshot()
        info_card = self._mini_card("Motor Bilgisi")
        for lbl, val in (
            ("Giriş cihazı", snap.input_name),
            ("Çıkış cihazı", snap.output_name),
            ("Örnekleme hızı", f"{snap.sample_rate:.0f} Hz"),
            ("FFT gecikmesi", f"{snap.latency_ms:.1f} ms"),
        ):
            row = QHBoxLayout()
            row.addWidget(self._label(lbl, TEXT2))
            row.addStretch()
            v = self._label(val, ACCENT)
            v.setObjectName("infoVal")
            row.addWidget(v)
            info_card.layout().addLayout(row)
        lay.addWidget(info_card)

        # Stereo routing
        route_card = self._mini_card("Stereo Yönlendirme")
        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(self._label("Sol hoparlör", TEXT2), 0, 0)
        self.left_output_combo = QComboBox()
        self.left_output_combo.setObjectName("settingsCombo")
        grid.addWidget(self.left_output_combo, 0, 1)

        grid.addWidget(self._label("Sağ hoparlör", TEXT2), 1, 0)
        self.right_output_combo = QComboBox()
        self.right_output_combo.setObjectName("settingsCombo")
        grid.addWidget(self.right_output_combo, 1, 1)

        btn_row = QHBoxLayout()
        self.stereo_switch = QPushButton("L/R Ayrı")
        self.stereo_switch.setCheckable(True)
        self.stereo_switch.setChecked(self.engine.left_output_device_index != self.engine.right_output_device_index)
        self.stereo_switch.setObjectName("pillButton")
        self.stereo_switch.toggled.connect(self._stereo_routing_changed)
        btn_row.addWidget(self.stereo_switch)

        self.swap_button = QPushButton("⇄  L/R Değiştir")
        self.swap_button.setObjectName("pillButton")
        self.swap_button.clicked.connect(self._swap_speakers)
        btn_row.addWidget(self.swap_button)

        save_btn = QPushButton("💾  Yönlendirmeyi Kaydet")
        save_btn.setObjectName("accentButton")
        save_btn.clicked.connect(self._save_routing)
        btn_row.addWidget(save_btn)
        btn_row.addStretch()

        route_card.layout().addLayout(grid)
        route_card.layout().addLayout(btn_row)
        lay.addWidget(route_card)

        # Görünüm & Tema Renkleri
        theme_card = self._mini_card("Arayüz Tema Rengi")
        theme_row = QHBoxLayout()
        theme_row.setSpacing(12)
        theme_row.addWidget(self._label("Tema Paleti:", TEXT2))

        self.dialog_theme_swatches: dict[str, QPushButton] = {}
        for t_name in THEMES.keys():
            btn = QPushButton()
            btn.setFixedSize(26, 26)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(f"Tema: {t_name}")
            btn.clicked.connect(lambda _, name=t_name: self._set_theme(name))
            theme_row.addWidget(btn)
            self.dialog_theme_swatches[t_name] = btn

        theme_row.addStretch()
        theme_card.layout().addLayout(theme_row)
        self._refresh_dialog_theme_swatches()
        lay.addWidget(theme_card)

        lay.addStretch()

        self._refresh_device_combos()
        return w

    # ── Speaker EQ tab ────────────────────────────────────────────────────────
    def _build_speaker_eq_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(16)

        desc = self._label("Her frekans bandı için sol ve sağ hoparlör kazancını ayrı ayrı ayarlayın. "
                           "Bu ayarlar donanıma özeldir ve kalıcı olarak kaydedilir.", TEXT2)
        desc.setWordWrap(True)
        lay.addWidget(desc)

        grid = QGridLayout()
        grid.setSpacing(8)
        grid.addWidget(self._label("Band", TEXT3), 0, 0)
        grid.addWidget(self._label("Sol (L)", ACCENT, bold=True, align=Qt.AlignmentFlag.AlignCenter), 0, 1)
        grid.addWidget(self._label("Sağ (R)", ACCENT_DIM, bold=True, align=Qt.AlignmentFlag.AlignCenter), 0, 2)
        grid.addWidget(self._label("dB (L)", TEXT3, align=Qt.AlignmentFlag.AlignCenter), 0, 3)
        grid.addWidget(self._label("dB (R)", TEXT3, align=Qt.AlignmentFlag.AlignCenter), 0, 4)

        self.speaker_sliders: dict[tuple[int, int], QSlider] = {}
        self.speaker_val_labels: dict[tuple[int, int], QLabel] = {}
        FREQ_LABELS = ("60 Hz", "250 Hz", "1 kHz", "4 kHz", "12 kHz")

        spk_gains = self.engine._speaker_gains

        for bi, (band, freq) in enumerate(zip(BANDS, FREQ_LABELS)):
            freq_lbl = self._label(freq, TEXT3)
            band_lbl = self._label(band, TEXT, bold=True)
            vbox = QVBoxLayout()
            vbox.setSpacing(1)
            vbox.addWidget(band_lbl)
            vbox.addWidget(freq_lbl)
            container = QWidget()
            container.setLayout(vbox)
            grid.addWidget(container, bi + 1, 0)

            for ch in (0, 1):
                sl = QSlider(Qt.Orientation.Horizontal)
                sl.setRange(-120, 120)
                init_val = int(round(float(spk_gains[ch][bi]) * 10))
                sl.setValue(init_val)
                sl.setObjectName("speakerSliderL" if ch == 0 else "speakerSliderR")
                sl.valueChanged.connect(
                    lambda v, c=ch, b=bi: self._speaker_gain_changed(c, b, v)
                )
                grid.addWidget(sl, bi + 1, ch + 1)
                self.speaker_sliders[(ch, bi)] = sl

                val_lbl = self._label(f"{init_val/10:+.1f}", ACCENT if ch == 0 else ACCENT_DIM,
                                      align=Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(val_lbl, bi + 1, ch + 3)
                self.speaker_val_labels[(ch, bi)] = val_lbl

        lay.addLayout(grid)

        row = QHBoxLayout()
        reset_btn = QPushButton("↺  Sıfırla")
        reset_btn.setObjectName("dangerButton")
        reset_btn.clicked.connect(self._reset_speaker_eq)
        row.addWidget(reset_btn)

        row.addStretch()

        save_btn = QPushButton("💾  Speaker EQ'yu Kaydet")
        save_btn.setObjectName("accentButton")
        save_btn.clicked.connect(self._save_speaker_settings)
        row.addWidget(save_btn)

        lay.addLayout(row)
        lay.addStretch()
        return w

    # ── Speaker DSP tab ───────────────────────────────────────────────────────
    def _build_speaker_dsp_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("settingsScroll")
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        lay = QVBoxLayout(scroll_content)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(14)

        desc = self._label(
            "Hoparlör bazlı renk ve kalibrasyon ayarları. Sol hoparlörden ses daha az geliyorsa "
            "Sol seviye çarpanını yükseltebilir veya hoparlör DSP netliğini artırabilirsiniz.",
            TEXT2
        )
        desc.setWordWrap(True)
        lay.addWidget(desc)

        self.speaker_dsp_sliders: dict[tuple[int, str], QSlider] = {}
        DSP_DEFS = [
            ("clarity",    "Clarity",    "Netlik / Hava Frekansları"),
            ("bass_boost", "Bass Boost", "Düşük Frekans Dolgunluğu"),
        ]
        CH_LABELS = [("Sol Hoparlör (L)", ACCENT), ("Sağ Hoparlör (R)", ACCENT_DIM)]

        # Yan yana esnek kart düzeni
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)

        for ch, (ch_name, ch_color) in enumerate(CH_LABELS):
            card = self._mini_card(ch_name)
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            card.layout().setSpacing(8)
            for key, title, desc_txt in DSP_DEFS:
                col = QVBoxLayout()
                header = QHBoxLayout()
                header.addWidget(self._label(title, ch_color, bold=True))
                header.addStretch()
                init_p = int(round(self.engine._speaker_features[ch].get(key, 0.0) * 100))
                val_lbl = self._label(f"{init_p} %", TEXT2)
                val_lbl.setObjectName(f"dspVal_{ch}_{key}")
                header.addWidget(val_lbl)
                col.addLayout(header)
                col.addWidget(self._label(desc_txt, TEXT3))
                sl = QSlider(Qt.Orientation.Horizontal)
                sl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                sl.setRange(0, 100)
                sl.setValue(init_p)
                sl.valueChanged.connect(
                    lambda v, c=ch, k=key, lbl=val_lbl: self._speaker_dsp_changed(c, k, v, lbl)
                )
                col.addWidget(sl)
                card.layout().addLayout(col)
                self.speaker_dsp_sliders[(ch, key)] = sl
            cards_row.addWidget(card, 1)

        lay.addLayout(cards_row)

        # Kalibrasyon
        cal_card = self._mini_card("Hoparlör Kalibrasyonu (Seviye Dengeleme)")
        cal_grid = QGridLayout()
        cal_grid.setSpacing(10)
        cal_grid.setColumnStretch(0, 0)
        cal_grid.setColumnStretch(1, 1)
        cal_grid.setColumnStretch(2, 0)

        left_cal = int(round(self.engine._calibration[0] * 100))
        right_cal = int(round(self.engine._calibration[1] * 100))

        cal_grid.addWidget(self._label("Sol hoparlör kazancı", TEXT2), 0, 0)
        self.cal_left = QSlider(Qt.Orientation.Horizontal)
        self.cal_left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.cal_left.setRange(50, 200)
        self.cal_left.setValue(left_cal)
        self.cal_left.valueChanged.connect(self._calibration_changed)
        cal_grid.addWidget(self.cal_left, 0, 1)
        self.cal_left_lbl = self._label(f"{left_cal/100:.2f}×", ACCENT)
        cal_grid.addWidget(self.cal_left_lbl, 0, 2)

        cal_grid.addWidget(self._label("Sağ hoparlör kazancı", TEXT2), 1, 0)
        self.cal_right = QSlider(Qt.Orientation.Horizontal)
        self.cal_right.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.cal_right.setRange(50, 200)
        self.cal_right.setValue(right_cal)
        self.cal_right.valueChanged.connect(self._calibration_changed)
        cal_grid.addWidget(self.cal_right, 1, 1)
        self.cal_right_lbl = self._label(f"{right_cal/100:.2f}×", WARN)
        cal_grid.addWidget(self.cal_right_lbl, 1, 2)

        cal_card.layout().addLayout(cal_grid)
        lay.addWidget(cal_card)

        row = QHBoxLayout()
        row.addStretch()
        save_btn = QPushButton("💾  Speaker DSP & Kalibrasyonu Kaydet")
        save_btn.setObjectName("accentButton")
        save_btn.clicked.connect(self._save_speaker_settings)
        row.addWidget(save_btn)
        lay.addLayout(row)

        lay.addStretch()
        scroll.setWidget(scroll_content)
        return scroll

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _speaker_gain_changed(self, ch: int, bi: int, v: int) -> None:
        db = v / 10.0
        self.engine.set_speaker_gain(ch, bi, db)
        self.speaker_val_labels[(ch, bi)].setText(f"{db:+.1f}")

    def _speaker_dsp_changed(self, ch: int, key: str, v: int, lbl: QLabel) -> None:
        self.engine.set_speaker_feature(ch, key, v / 100.0)
        lbl.setText(f"{v} %")

    def _calibration_changed(self) -> None:
        left  = self.cal_left.value()  / 100.0
        right = self.cal_right.value() / 100.0
        self.engine.set_speaker_calibration(left, right)
        self.cal_left_lbl.setText(f"{left:.2f}×")
        self.cal_right_lbl.setText(f"{right:.2f}×")
        if self.parent() and hasattr(self.parent(), "_sync_calibration_from_engine"):
            self.parent()._sync_calibration_from_engine()

    def _save_speaker_settings(self) -> None:
        self.engine.save_speaker_settings()
        QMessageBox.information(self, "AuraLab", "Hoparlör ayarları başarıyla kaydedildi!")

    def _save_routing(self) -> None:
        self.engine.save_speaker_settings()
        QMessageBox.information(self, "AuraLab", "Yönlendirme ayarları kaydedildi!")

    def _reset_speaker_eq(self) -> None:
        for (ch, bi), sl in self.speaker_sliders.items():
            sl.setValue(0)
        self.engine.save_speaker_settings()

    def _refresh_device_combos(self) -> None:
        for combo, selected in (
            (self.left_output_combo,  self.engine.left_output_device_index),
            (self.right_output_combo, self.engine.right_output_device_index),
        ):
            combo.blockSignals(True)
            combo.clear()
            for index, info in self.engine.output_devices:
                combo.addItem(info["name"], index)
            row = next((r for r, (idx, _) in enumerate(self.engine.output_devices) if idx == selected), 0)
            combo.setCurrentIndex(row)
            combo.blockSignals(False)

    def _stereo_routing_changed(self, separate: bool) -> None:
        try:
            self.stereo_switch.setText("L/R Ayrı" if separate else "Tek Çıkış")
            left  = self.left_output_combo.currentData()
            right = self.right_output_combo.currentData()
            if not separate:
                if left is not None:
                    try:
                        self.engine.switch_output_device(int(left))
                    except Exception as e:
                        QMessageBox.warning(self, "AuraLab", f"Çıkış değiştirilemedi:\n{e}")
                return
            if left is not None and right is not None:
                try:
                    left_name  = next(i["name"] for idx, i in self.engine.output_devices if idx == left)
                    right_name = next(i["name"] for idx, i in self.engine.output_devices if idx == right)
                    self.audio_manager.sync_to_physical(left_name)
                    self.audio_manager.sync_to_physical(right_name)
                except Exception:
                    pass
                try:
                    self.engine.set_channel_outputs(int(left), int(right))
                    if self.engine._last_error:
                        QMessageBox.warning(self, "AuraLab", f"Ayrık çıkış uyarısı:\n{self.engine._last_error}")
                except Exception as e:
                    QMessageBox.warning(self, "AuraLab", f"Ayrık çıkış başlatılamadı:\n{e}")
        except Exception as err:
            QMessageBox.warning(self, "AuraLab", f"Yönlendirme hatası:\n{err}")

    def _swap_speakers(self) -> None:
        if self.stereo_switch.isChecked():
            left  = self.left_output_combo.currentData()
            right = self.right_output_combo.currentData()
            if left is None or right is None:
                return
            self.left_output_combo.blockSignals(True)
            self.right_output_combo.blockSignals(True)
            li = self.left_output_combo.currentIndex()
            ri = self.right_output_combo.currentIndex()
            self.left_output_combo.setCurrentIndex(ri)
            self.right_output_combo.setCurrentIndex(li)
            self.left_output_combo.blockSignals(False)
            self.right_output_combo.blockSignals(False)
            self._stereo_routing_changed(True)
        else:
            new_swap = not self.engine._lr_swap
            self.engine.set_lr_swap(new_swap)
            self.engine.save_speaker_settings()
            QMessageBox.information(
                self, "AuraLab",
                f"Sol/Sağ Kanal Takası: {'Aktif' if new_swap else 'Kapalı'}"
            )

    @staticmethod
    def _label(text: str, color: str = TEXT, bold: bool = False,
               align: Qt.AlignmentFlag | None = None) -> QLabel:
        lbl = QLabel(text)
        style = f"color:{color}; font-size:12px;"
        if bold:
            style += " font-weight:700;"
        lbl.setStyleSheet(style)
        if align:
            lbl.setAlignment(align)
        return lbl

    def _mini_card(self, title: str) -> QFrame:
        card = QFrame()
        card.setObjectName("miniCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(8)
        lbl = QLabel(title)
        lbl.setObjectName("miniCardTitle")
        lay.addWidget(lbl)
        return card

    def _set_theme(self, name: str) -> None:
        parent = self.parent()
        if parent is not None and hasattr(parent, "_change_theme"):
            parent._change_theme(name)
        else:
            _apply_accent(name)
            _save_theme(name)
        self._apply_dialog_theme()
        self._refresh_dialog_theme_swatches()

    def _refresh_dialog_theme_swatches(self) -> None:
        if not hasattr(self, "dialog_theme_swatches"):
            return
        cur_theme = _load_theme()
        for t_name, btn in self.dialog_theme_swatches.items():
            c = THEMES[t_name]["ACCENT"]
            is_active = (t_name == cur_theme)
            if is_active:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {c};
                        border: 2px solid #ffffff;
                        border-radius: 13px;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {c};
                        border: 1px solid rgba(255, 255, 255, 0.35);
                        border-radius: 13px;
                    }}
                    QPushButton:hover {{
                        border: 2px solid rgba(255, 255, 255, 0.85);
                    }}
                """)

    def _apply_dialog_theme(self) -> None:
        self.setStyleSheet(f"""
            QDialog {{ background-color:{BG}; }}
            QTabWidget::pane {{ border:1px solid {BORDER}; background:{BG2}; border-radius:10px; }}
            QTabBar::tab {{
                background:{BG3}; color:{TEXT2}; padding:10px 22px;
                border-top-left-radius:8px; border-top-right-radius:8px;
                font-weight:600; font-size:12px; margin-right:4px;
            }}
            QTabBar::tab:selected {{ background:{BG2}; color:{ACCENT}; border-top:2px solid {ACCENT}; }}
            #miniCard {{ background:{BG3}; border:1px solid {BORDER}; border-radius:8px; }}
            #miniCardTitle {{ color:{TEXT}; font-weight:700; font-size:13px; }}
            #settingsCombo {{
                background:{BG2}; border:1px solid {BORDER}; border-radius:6px;
                color:{TEXT}; padding:6px 12px; font-size:12px; min-width:240px;
            }}
            #settingsCombo:hover {{ border-color:{BORDER_HV}; }}
            #pillButton {{
                background:{BG2}; border:1px solid {BORDER}; border-radius:6px;
                color:{TEXT}; padding:6px 16px; font-weight:600; font-size:12px;
            }}
            #pillButton:checked {{ background:{ACCENT_BG}; border-color:{ACCENT}; color:{ACCENT}; }}
            #accentButton {{
                background:{ACCENT}; color:{BG}; border:none; border-radius:6px;
                padding:8px 18px; font-weight:700; font-size:12px;
            }}
            #accentButton:hover {{ background:#a8f84a; }}
            #dangerButton {{
                background:#2a1010; color:#ff6060; border:1px solid #5a2020;
                border-radius:6px; padding:6px 14px; font-size:12px;
            }}
            QSlider::groove:horizontal {{ height:4px; background:{BORDER}; border-radius:2px; }}
            QSlider::sub-page:horizontal {{ background:{ACCENT}; border-radius:2px; }}
            QSlider::handle:horizontal {{
                background:#fff; border:2px solid {ACCENT}; width:14px;
                margin:-5px 0; border-radius:7px;
            }}
            #speakerSliderL::sub-page:horizontal {{ background:{ACCENT}; }}
            #speakerSliderL::handle:horizontal {{ border-color:{ACCENT}; }}
            #speakerSliderR::sub-page:horizontal {{ background:{ACCENT_DIM}; }}
            #speakerSliderR::handle:horizontal {{ border-color:{ACCENT_DIM}; }}
            #infoVal {{ color:{ACCENT}; font-weight:700; }}
            QLabel {{ color:{TEXT}; }}
        """)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ──────────────────────────────────────────────────────────────────────────────
class AutoEQWindow(QMainWindow):
    def __init__(self, engine: AudioEngine, audio_manager: SystemAudioManager) -> None:
        super().__init__()
        self.engine = engine
        self.audio_manager = audio_manager
        self._last_system_volume: tuple[float, bool] | None = None
        self.sliders: list[QSlider] = []
        self.value_labels: list[QLabel] = []
        self.feature_sliders: dict[str, QSlider] = {}
        self.feature_val_labels: dict[str, QLabel] = {}
        self._updating_ui = False
        self.spotify = SpotifyIntegration()
        self._last_spotify_track_id: str | None = None
        self._current_spotify_genre = ""
        self._last_manual_eq_time: float = 0.0
        self._MANUAL_EQ_GRACE = 30.0
        self._settings_dialog: SettingsDialog | None = None
        self._current_theme = _load_theme()

        # Sistem güncel master sesini anında oku ve başlarken eşitle (anilik/sıçrama olmaması için)
        try:
            init_vol, init_muted = self.audio_manager.current_master()
            self._last_system_volume = (round(init_vol, 3), init_muted)
            self.engine.set_volume(0.0 if init_muted else init_vol)
        except Exception:
            pass

        self.setWindowTitle("AuraLab — Akıllı Ses Stüdyosu")
        self.setMinimumSize(1080, 680)
        self.resize(1220, 750)
        self._build_ui()
        self._apply_theme()
        self._setup_tray()
        self.engine.start()

        # İlk açılışta DSP slider'larını Flat veya varsayılan preset değerlerine senkronize et
        self._sync_dsp_from_engine()
        self._update_routing_button_state(self.audio_manager.is_cable_default())

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(100)

        # Spectrum timer — 30 FPS (33ms) — gözle 60fps'ten ayırt edilemez, CPU yükü yarı yarıya
        self.spectrum_timer = QTimer(self)
        self.spectrum_timer.timeout.connect(self._refresh_spectrum)
        self.spectrum_timer.start(33)  # 30 FPS

        self.system_timer = QTimer(self)
        self.system_timer.timeout.connect(self._refresh_system_audio)
        self.system_timer.start(250)

        self.spotify_timer = QTimer(self)
        self.spotify_timer.timeout.connect(self._refresh_spotify)
        self.spotify_timer.start(4000)
        QTimer.singleShot(200, self._refresh_spotify)

        # Canlı durum noktası yanıp sönme
        self._dot_state = True
        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._blink_dot)
        self._dot_timer.start(1200)

    # ── Tray ──────────────────────────────────────────────────────────────────
    def _setup_tray(self) -> None:
        self.tray = QSystemTrayIcon(self)
        self.tray.setToolTip("AuraLab")
        icon = create_auralab_icon(ACCENT)
        self.tray.setIcon(icon)
        menu = QMenu(self)
        menu.addAction("AuraLab'ı Aç").triggered.connect(self._show_from_tray)
        menu.addSeparator()
        menu.addAction("Yönlendirmeyi Değiştir (Hoparlör / AuraLab)").triggered.connect(self._toggle_audio_routing)
        menu.addAction("Çıkış & Sesi Geri Al").triggered.connect(self._exit_clean)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda r: self._show_from_tray()
            if r == QSystemTrayIcon.ActivationReason.DoubleClick else None
        )
        self.tray.show()

    def _show_from_tray(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def _exit_clean(self) -> None:
        self.close()

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("root")
        self.setCentralWidget(central)

        outer_lay = QVBoxLayout(central)
        outer_lay.setContentsMargins(0, 0, 0, 0)
        outer_lay.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("mainScroll")
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        scroll_content.setObjectName("scrollContent")
        main_lay = QVBoxLayout(scroll_content)
        main_lay.setContentsMargins(18, 10, 18, 10)
        main_lay.setSpacing(8)

        # 1. Header
        main_lay.addWidget(self._build_header())

        # 2. Hero Bar (Spotify + Status + Power)
        main_lay.addWidget(self._build_hero())

        # 3. Spectrum Analyzer
        main_lay.addWidget(self._build_spectrum_card())

        # 4. Controls Row: Equalizer + Routing/DSP
        main_lay.addLayout(self._build_controls_row(), 1)

        scroll.setWidget(scroll_content)
        outer_lay.addWidget(scroll)

        # 5. Footer — scroll area DIŞINDA, her zaman görünür
        footer_wrap = QWidget()
        footer_wrap.setObjectName("footerWrap")
        fw_lay = QVBoxLayout(footer_wrap)
        fw_lay.setContentsMargins(18, 0, 18, 6)
        fw_lay.setSpacing(0)
        fw_lay.addWidget(self._build_footer())
        outer_lay.addWidget(footer_wrap)

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        logo = QLabel("AURALAB")
        logo.setObjectName("logoLabel")
        lay.addWidget(logo)

        ver = QLabel("v2.3")
        ver.setObjectName("verBadge")
        lay.addWidget(ver)

        # Canlı durum göstergesi
        self.live_indicator = QLabel("⬤  LIVE")
        self.live_indicator.setObjectName("liveIndicator")
        lay.addWidget(self.live_indicator)

        lay.addStretch()

        # Gelişmiş Ayarlar butonu
        settings_btn = QPushButton("⚙  Gelişmiş Ayarlar")
        settings_btn.setObjectName("headerButton")
        settings_btn.setToolTip("Hoparlör EQ, ayrıntılı kanal kalibrasyonu ve motor ayarları")
        settings_btn.clicked.connect(self._open_settings)
        lay.addWidget(settings_btn)

        return w

    # ── Hero bar ──────────────────────────────────────────────────────────────
    def _build_hero(self) -> QFrame:
        hero = QFrame()
        hero.setObjectName("heroBar")
        lay = QHBoxLayout(hero)
        lay.setContentsMargins(16, 6, 16, 6)
        lay.setSpacing(14)

        info = QVBoxLayout()
        info.setSpacing(2)
        self.track_label = QLabel("Spotify  /  bekleniyor...")
        self.track_label.setObjectName("trackLabel")
        info.addWidget(self.track_label)

        sub_row = QHBoxLayout()
        sub_row.setSpacing(12)
        self.genre_detect_label = QLabel("Algılanan tür:  —")
        self.genre_detect_label.setObjectName("genreDetect")
        sub_row.addWidget(self.genre_detect_label)

        self.mode_label = QLabel("EQ ve uzamsal DSP aktif")
        self.mode_label.setObjectName("modeLabel")
        sub_row.addWidget(self.mode_label)
        sub_row.addStretch()
        info.addLayout(sub_row)

        lay.addLayout(info, 1)

        # Latency pill
        self.tech_label = QLabel("48000 Hz  •  42.7 ms")
        self.tech_label.setObjectName("techPill")
        self.tech_label.setToolTip("Örnekleme Hızı: 48000 Hz | Ses İşleme Tampon Gecikmesi: 42.7 ms")
        lay.addWidget(self.tech_label, 0, Qt.AlignmentFlag.AlignVCenter)

        # Master Power / Routing Toggle (AuraLab Motoru <-> Doğrudan Hoparlör)
        self.toggle = QPushButton("⏻")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setObjectName("powerToggle")
        self.toggle.setFixedSize(44, 44)
        self.toggle.setToolTip("AuraLab Ses Motoru Aktif (VB-Cable)\nTıklayarak doğrudan hoparlöre geçiş yapabilirsiniz.")
        self.toggle.clicked.connect(self._toggle_audio_routing)
        lay.addWidget(self.toggle, 0, Qt.AlignmentFlag.AlignVCenter)

        return hero

    # ── Spectrum ──────────────────────────────────────────────────────────────
    def _build_spectrum_card(self) -> QFrame:
        # Özel başlık: standart _card yerine elle oluştur
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 6, 14, 8)
        lay.setSpacing(4)

        # Başlık satırı
        hdr = QHBoxLayout()
        title_lbl = QLabel("◈  SPECTRUM")
        title_lbl.setObjectName("spectrumTitle")
        hdr.addWidget(title_lbl)
        hdr.addStretch()
        range_lbl = QLabel("20 Hz  —  20 kHz")
        range_lbl.setObjectName("spectrumRange")
        hdr.addWidget(range_lbl)
        lay.addLayout(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("cardSep")
        lay.addWidget(sep)

        self.canvas = SpectrumCanvas()
        lay.addWidget(self.canvas)
        return card

    # ── Controls row: Equalizer (sol) | Routing + DSP (sağ) ────────────────────
    def _build_controls_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        # Sol sütun: EQUALIZER (Genre ve Preset toolbar ile)
        row.addWidget(self._build_eq_card(), 6)
        # Sağ sütun: Stereo Yönlendirme / Hoparlör Kalibrasyonu + DSP Efektleri
        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        right_col.addWidget(self._build_routing_card())
        right_col.addWidget(self._build_dsp_card())
        row.addLayout(right_col, 5)
        return row

    # ── EQUALIZER CARD ────────────────────────────────────────────────────────
    def _build_eq_card(self) -> QFrame:
        card = self._card("EQUALİZER & PROFİLLER")
        card.setMinimumWidth(520)
        lay = card.layout()
        lay.setContentsMargins(14, 4, 14, 8)
        lay.setSpacing(4)

        # ── Toolbar: Genre EQ + Preset + Reset (Genre önce!) ──────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        # Genre EQ Grubu (önce!)
        toolbar.addWidget(QLabel("Genre EQ:"))
        self.genre_combo = QComboBox()
        self.genre_combo.setObjectName("mainCombo")
        self.genre_combo.setEditable(True)
        self.genre_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.genre_combo.addItems(GENRE_PROFILES.keys())
        self.genre_combo.currentTextChanged.connect(self._genre_profile_changed)
        toolbar.addWidget(self.genre_combo)

        self.save_genre_btn = QPushButton("💾 Kaydet")
        self.save_genre_btn.setObjectName("compactSaveBtn")
        self.save_genre_btn.setToolTip("Mevcut EQ ve DSP değerlerini seçili müzik türüne kalıcı olarak kaydet")
        self.save_genre_btn.clicked.connect(self._save_genre_profile)
        toolbar.addWidget(self.save_genre_btn)

        toolbar.addSpacing(8)

        # Preset Grubu (sonra!)
        toolbar.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.setObjectName("mainCombo")
        self.preset_combo.addItems(PRESETS.keys())
        self.preset_combo.currentTextChanged.connect(self._preset_changed)
        toolbar.addWidget(self.preset_combo)

        self.save_preset_btn = QPushButton("💾 Kaydet")
        self.save_preset_btn.setObjectName("compactSaveBtn")
        self.save_preset_btn.setToolTip("Mevcut EQ kazançlarını ve DSP (Clarity, Bass vb.) değerlerini bu presete kaydet")
        self.save_preset_btn.clicked.connect(self._save_preset)
        toolbar.addWidget(self.save_preset_btn)

        toolbar.addStretch()

        self.reset_btn = QPushButton("↺ EQ Sıfırla")
        self.reset_btn.setObjectName("ghostButton")
        self.reset_btn.clicked.connect(self._reset_eq)
        toolbar.addWidget(self.reset_btn)

        lay.addLayout(toolbar)

        # ── 5 Bant Slider Grid ────────────────────────────────────────
        FREQ_LABELS = ("60 Hz", "250 Hz", "1 kHz", "4 kHz", "12 kHz")
        grid = QGridLayout()
        grid.setContentsMargins(0, 4, 0, 2)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(2)

        for i, (band, freq) in enumerate(zip(BANDS, FREQ_LABELS)):
            # Frekans (Üstte)
            fl = QLabel(freq)
            fl.setObjectName("freqLabel")
            fl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(fl, 0, i)

            # Dikey Slider — responsive, uzun
            sl = QSlider(Qt.Orientation.Vertical)
            sl.setRange(-120, 120)
            sl.setValue(0)
            sl.setMinimumHeight(200)
            sl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            sl.setObjectName("eqSlider")
            sl.valueChanged.connect(lambda v, idx=i: self._gain_changed(idx, v / 10.0))
            self.sliders.append(sl)
            grid.addWidget(sl, 1, i, Qt.AlignmentFlag.AlignCenter)

            # Bant adı
            bl = QLabel(band)
            bl.setObjectName("bandLabel")
            bl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(bl, 2, i, Qt.AlignmentFlag.AlignCenter)

            # Değer etiketi
            val = QLabel("  0.0 dB")
            val.setObjectName("gainValue")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.value_labels.append(val)
            grid.addWidget(val, 3, i, Qt.AlignmentFlag.AlignCenter)

        grid.setRowStretch(1, 1)
        lay.addLayout(grid, 1)
        return card

    # ── STEREO ROUTING CARD (Ana ekranda!) ────────────────────────────────────
    def _build_routing_card(self) -> QFrame:
        card = self._card("STEREO YÖNLENDİRME")
        card.setMinimumWidth(440)
        lay = card.layout()
        lay.setContentsMargins(14, 8, 14, 10)
        lay.setSpacing(8)

        # Çıkış Aygıtı Seçimi
        dev_row = QHBoxLayout()
        dev_row.setSpacing(8)
        dev_row.addWidget(QLabel("Hoparlör:"))
        self.main_output_combo = QComboBox()
        self.main_output_combo.setObjectName("mainCombo")
        self._refresh_main_device_combo()
        self.main_output_combo.currentIndexChanged.connect(self._main_output_changed)
        dev_row.addWidget(self.main_output_combo, 1)

        # L/R Ayrı Toggle
        self.lr_separate_toggle = QPushButton("🔀 L/R Ayrık")
        self.lr_separate_toggle.setCheckable(True)
        is_sep = bool(self.engine._speaker_settings.get("is_separate", False)) and (
            self.engine.left_output_device_index != self.engine.right_output_device_index
        )
        self.lr_separate_toggle.setChecked(is_sep)
        self.lr_separate_toggle.setObjectName("lrSeparateButton")
        self.lr_separate_toggle.toggled.connect(self._toggle_separate_routing)
        dev_row.addWidget(self.lr_separate_toggle)

        self.swap_btn = QPushButton("⇄ Swap")
        self.swap_btn.setObjectName("ghostButton")
        self.swap_btn.setToolTip("Sol ve sağ hoparlör kanallarını yer değiştir")
        self.swap_btn.clicked.connect(self._swap_channels)
        dev_row.addWidget(self.swap_btn)

        lay.addLayout(dev_row)

        # Ayrık Hoparlör Seçim Paneli — HER ZAMAN görünür; L/R Ayrık kapalıyken soluk/tıklanamaz
        self.separate_panel = QWidget()
        sep_lay = QHBoxLayout(self.separate_panel)
        sep_lay.setContentsMargins(0, 0, 0, 0)
        sep_lay.setSpacing(8)
        self._sep_lbl_left = QLabel("Sol (L):")
        sep_lay.addWidget(self._sep_lbl_left)
        self.sep_left_combo = QComboBox()
        self.sep_left_combo.setObjectName("mainCombo")
        sep_lay.addWidget(self.sep_left_combo, 1)

        self._sep_lbl_right = QLabel("Sağ (R):")
        sep_lay.addWidget(self._sep_lbl_right)
        self.sep_right_combo = QComboBox()
        self.sep_right_combo.setObjectName("mainCombo")
        sep_lay.addWidget(self.sep_right_combo, 1)
        # Cihaz combo'larını her zaman doldur (enabled/disabled olsa bile içerik görünsün)
        for combo, sel in ((self.sep_left_combo, self.engine.left_output_device_index),
                           (self.sep_right_combo, self.engine.right_output_device_index)):
            combo.blockSignals(True)
            combo.clear()
            for index, info in self.engine.output_devices:
                combo.addItem(info["name"], index)
            r = next((i for i, (idx, _) in enumerate(self.engine.output_devices) if idx == sel), 0)
            combo.setCurrentIndex(r)
            combo.blockSignals(False)
        self.separate_panel.setVisible(True)  # HER ZAMAN görünür
        self._set_sep_panel_enabled(is_sep)   # Sadece etkinlik durumu değişir
        lay.addWidget(self.separate_panel)

        return card

    # ── DSP CARD ──────────────────────────────────────────────────────────────
    def _build_dsp_card(self) -> QFrame:
        card = self._card("SES İŞLEME (DSP)")
        card.setMinimumWidth(440)
        lay = card.layout()
        lay.setContentsMargins(14, 6, 14, 8)
        lay.setSpacing(4)

        dsp_grid = QGridLayout()
        dsp_grid.setContentsMargins(0, 4, 0, 4)
        dsp_grid.setHorizontalSpacing(12)
        dsp_grid.setVerticalSpacing(6)
        dsp_grid.setColumnStretch(0, 0)
        dsp_grid.setColumnStretch(1, 1)
        dsp_grid.setColumnStretch(2, 0)

        DSP_ITEMS = [
            ("clarity",       "Clarity",       "Yüksek frekans netliği"),
            ("bass_boost",    "Bass Boost",    "Dinamik bas güçlendirme"),
            ("dynamic_boost", "Dynamic Boost", "Akıllı dinamik sıkıştırma"),
            ("ambience",      "Ambience",      "Oda yansıması ve mekan"),
            ("surround",      "Surround",      "Geniş stereo sahne"),
        ]

        for row_idx, (key, title, tip) in enumerate(DSP_ITEMS):
            t = QLabel(title)
            t.setObjectName("dspTitle")
            t.setMinimumWidth(105)
            t.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            dsp_grid.addWidget(t, row_idx, 0)

            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, 100)
            sl.setObjectName("dspSlider")
            sl.setToolTip(tip)
            dsp_grid.addWidget(sl, row_idx, 1)

            v = QLabel("0 %")
            v.setObjectName("dspValue")
            v.setFixedWidth(45)
            v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            dsp_grid.addWidget(v, row_idx, 2)

            sl.valueChanged.connect(lambda val, k=key, lbl=v: self._on_dsp_slider_changed(k, val, lbl))

            self.feature_sliders[key] = sl
            self.feature_val_labels[key] = v

        lay.addLayout(dsp_grid)
        return card

    # ── Footer ────────────────────────────────────────────────────────────────
    def _build_footer(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("footerBar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 6, 16, 6)
        lay.setSpacing(14)

        # Volume
        lay.addWidget(self._footer_label("🔊  Ses"))
        init_pct = round(self._last_system_volume[0] * 100) if self._last_system_volume else 100
        is_muted = self._last_system_volume[1] if self._last_system_volume else False

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 150)
        self.volume_slider.setValue(init_pct)
        self.volume_slider.setFixedWidth(150)
        self.volume_slider.setObjectName("footerSlider")
        self.volume_slider.valueChanged.connect(self._volume_changed)
        lay.addWidget(self.volume_slider)

        self.vol_label = QLabel(f"{init_pct} %")
        self.vol_label.setObjectName("footerVal")
        lay.addWidget(self.vol_label)

        self.mute_button = QPushButton("Sessiz ✕" if is_muted else "Sustur")
        self.mute_button.setCheckable(True)
        self.mute_button.setChecked(is_muted)
        self.mute_button.setObjectName("muteButton")
        self.mute_button.toggled.connect(self._mute_changed)
        lay.addWidget(self.mute_button)

        self._vsep(lay)

        # Balance
        lay.addWidget(self._footer_label("⇔  Denge"))
        self.balance_slider = QSlider(Qt.Orientation.Horizontal)
        self.balance_slider.setRange(-100, 100)
        self.balance_slider.setValue(int(round(self.engine._balance * 100)))
        self.balance_slider.setFixedWidth(130)
        self.balance_slider.setObjectName("footerSlider")
        self.balance_slider.valueChanged.connect(self._balance_changed)
        lay.addWidget(self.balance_slider)

        self.bal_label = QLabel("Merkez")
        self.bal_label.setObjectName("footerVal")
        lay.addWidget(self.bal_label)

        lay.addStretch()

        return bar

    # ── Card / label helpers ──────────────────────────────────────────────────
    def _card(self, title: str) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 6, 14, 8)
        lay.setSpacing(4)
        lbl = QLabel(title)
        lbl.setObjectName("cardTitle")
        lay.addWidget(lbl)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("cardSep")
        lay.addWidget(sep)
        return card

    @staticmethod
    def _footer_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("footerLabel")
        return lbl

    @staticmethod
    def _vsep(lay: QHBoxLayout) -> None:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setObjectName("vSep")
        lay.addWidget(sep)

    # ── Logic: Stereo Routing & Main Screen Devices ───────────────────────────
    def _refresh_main_device_combo(self) -> None:
        self.main_output_combo.blockSignals(True)
        self.main_output_combo.clear()
        selected = self.engine.output_device_index
        row_to_select = 0
        for r, (index, info) in enumerate(self.engine.output_devices):
            self.main_output_combo.addItem(info["name"], index)
            if index == selected:
                row_to_select = r
        self.main_output_combo.setCurrentIndex(row_to_select)
        self.main_output_combo.blockSignals(False)

    def _main_output_changed(self, idx: int) -> None:
        try:
            dev_idx = self.main_output_combo.currentData()
            if dev_idx is not None:
                self.engine.switch_output_device(int(dev_idx))
                self.mode_label.setText(f"Çıkış değiştirildi: {self.main_output_combo.currentText()}")
        except Exception as err:
            self.mode_label.setText(f"Çıkış seçilemedi: {err}")

    def _set_sep_panel_enabled(self, enabled: bool) -> None:
        """Ayrık hoparlör panel combo'larını etkinleştir/devre dışı bırak."""
        for w in (self.sep_left_combo, self.sep_right_combo,
                  self._sep_lbl_left, self._sep_lbl_right):
            w.setEnabled(enabled)

    def _toggle_separate_routing(self, checked: bool) -> None:
        try:
            self.lr_separate_toggle.setText("🔀 L/R Ayrık (Aktif)" if checked else "🔀 L/R Ayrık")
            self._set_sep_panel_enabled(checked)
            if checked:
                # Combo'ları yenile (içerik değişmiş olabilir)
                for combo, sel in ((self.sep_left_combo, self.engine.left_output_device_index),
                                   (self.sep_right_combo, self.engine.right_output_device_index)):
                    combo.blockSignals(True)
                    combo.clear()
                    for index, info in self.engine.output_devices:
                        combo.addItem(info["name"], index)
                    r = next((i for i, (idx, _) in enumerate(self.engine.output_devices) if idx == sel), 0)
                    combo.setCurrentIndex(r)
                    combo.blockSignals(False)
                try:
                    self.sep_left_combo.currentIndexChanged.disconnect()
                except Exception:
                    pass
                try:
                    self.sep_right_combo.currentIndexChanged.disconnect()
                except Exception:
                    pass
                self.sep_left_combo.currentIndexChanged.connect(self._apply_separate_channel_outputs)
                self.sep_right_combo.currentIndexChanged.connect(self._apply_separate_channel_outputs)
                self._apply_separate_channel_outputs()
            else:
                dev_idx = self.main_output_combo.currentData()
                if dev_idx is not None:
                    self.engine.switch_output_device(int(dev_idx))
                    self.engine.save_speaker_settings()
                    self.mode_label.setText(f"Birleşik stereo çıkış aktif: {self.main_output_combo.currentText()}")
        except Exception as err:
            self.mode_label.setText(f"Yönlendirme hatası: {err}")

    def _apply_separate_channel_outputs(self) -> None:
        try:
            left = self.sep_left_combo.currentData()
            right = self.sep_right_combo.currentData()
            if left is not None and right is not None:
                self.engine.set_channel_outputs(int(left), int(right))
                if self.engine._last_error:
                    self.mode_label.setText(f"⚠ {self.engine._last_error}")
                else:
                    self.mode_label.setText(f"L/R ayrı çıkış: Sol: {self.sep_left_combo.currentText()} | Sağ: {self.sep_right_combo.currentText()}")
        except Exception as err:
            self.mode_label.setText(f"Ayrık kanal hatası: {err}")
            try:
                dev_idx = self.main_output_combo.currentData()
                if dev_idx is not None:
                    self.engine.switch_output_device(int(dev_idx))
            except Exception:
                pass

    def _swap_channels(self) -> None:
        try:
            if self.lr_separate_toggle.isChecked():
                li = self.sep_left_combo.currentIndex()
                ri = self.sep_right_combo.currentIndex()
                self.sep_left_combo.blockSignals(True)
                self.sep_right_combo.blockSignals(True)
                self.sep_left_combo.setCurrentIndex(ri)
                self.sep_right_combo.setCurrentIndex(li)
                self.sep_left_combo.blockSignals(False)
                self.sep_right_combo.blockSignals(False)
                self._apply_separate_channel_outputs()
            else:
                # L/R Swap toggle in engine
                new_swap = not self.engine._lr_swap
                self.engine.set_lr_swap(new_swap)
                self.engine.save_speaker_settings()
                self.mode_label.setText(f"Sol/Sağ Kanal Takası: {'Aktif' if new_swap else 'Kapalı'}")
        except Exception as err:
            self.mode_label.setText(f"Kanal takas hatası: {err}")

    def _save_speaker_settings(self) -> None:
        self.engine.save_speaker_settings()
        self.mode_label.setText("✓ Hoparlör kalibrasyonu ve yönlendirme kaydedildi")
        if hasattr(self, "save_cal_btn") and self.save_cal_btn is not None:
            self.save_cal_btn.setText("✓ Kaydedildi!")
            QTimer.singleShot(1200, lambda: self.save_cal_btn.setText("💾 Hoparlör Ayarlarını Kaydet"))

    def _toggle_audio_routing(self) -> None:
        try:
            is_cable, name = self.audio_manager.toggle_routing()
            self._update_routing_button_state(is_cable)
            if is_cable:
                self.mode_label.setText("✓ AuraLab devrede (Varsayılan ses: VB-Cable)")
            else:
                self.mode_label.setText(f"✓ Doğrudan hoparlör devrede: {name}")
        except Exception as e:
            self.mode_label.setText(f"Yönlendirme hatası: {e}")

    _restore_audio_manual = _toggle_audio_routing

    def _update_routing_button_state(self, is_cable: bool) -> None:
        if not hasattr(self, "toggle"):
            return
        if is_cable:
            self.toggle.setChecked(True)
            self.toggle.setText("⏻")
            self.toggle.setToolTip("AuraLab Ses Motoru Aktif (VB-Cable)\nTıklayarak doğrudan fiziksel hoparlöre geçebilirsiniz.")
            self.toggle.setProperty("directSpeaker", "false")
        else:
            self.toggle.setChecked(False)
            self.toggle.setText("⏻")
            self.toggle.setToolTip("Doğrudan Hoparlör Aktif (AuraLab Devre Dışı)\nTıklayarak sesi AuraLab ses motoruna bağlayabilirsiniz.")
            self.toggle.setProperty("directSpeaker", "true")
        self.toggle.style().unpolish(self.toggle)
        self.toggle.style().polish(self.toggle)

    # ── Logic: DSP & Presets ──────────────────────────────────────────────────
    def _on_dsp_slider_changed(self, key: str, val: int, lbl: QLabel) -> None:
        lbl.setText(f"{val} %")
        if not self._updating_ui:
            self.engine.set_feature(key, val / 100.0)
            self._last_manual_eq_time = time.time()

    def _sync_dsp_from_engine(self) -> None:
        self._updating_ui = True
        feats = self.engine.get_features()
        for k, sl in self.feature_sliders.items():
            pct = int(round(feats.get(k, 0.0) * 100))
            sl.blockSignals(True)
            sl.setValue(pct)
            sl.blockSignals(False)
            if k in self.feature_val_labels:
                self.feature_val_labels[k].setText(f"{pct} %")
        self._updating_ui = False

    def _preset_changed(self, name: str) -> None:
        self._updating_ui = True
        dsp = self.engine.apply_preset(name)
        gains = self.engine.effective_preset_gains(name)
        for i, sl in enumerate(self.sliders):
            if i < len(gains):
                val = int(round(gains[i] * 10))
                sl.blockSignals(True)
                sl.setValue(val)
                sl.blockSignals(False)
                sign = "+" if val >= 0 else ""
                self.value_labels[i].setText(f"{sign}{val/10:.1f} dB")

        # DSP slider'larını o preset'in değerlerine güncelle!
        for k, sl in self.feature_sliders.items():
            pct = int(round(dsp.get(k, 0.0) * 100))
            sl.blockSignals(True)
            sl.setValue(pct)
            sl.blockSignals(False)
            if k in self.feature_val_labels:
                self.feature_val_labels[k].setText(f"{pct} %")

        self._updating_ui = False
        self._last_manual_eq_time = time.time()
        self.mode_label.setText(f"Preset uygulandı: {name} (EQ + DSP)")

    def _genre_profile_changed(self, profile: str) -> None:
        self._updating_ui = True
        dsp = self.engine.apply_genre(profile)
        gains = self.engine.effective_genre_gains(profile)
        for i, sl in enumerate(self.sliders):
            if i < len(gains):
                val = int(round(gains[i] * 10))
                sl.blockSignals(True)
                sl.setValue(val)
                sl.blockSignals(False)
                sign = "+" if val >= 0 else ""
                self.value_labels[i].setText(f"{sign}{val/10:.1f} dB")

        for k, sl in self.feature_sliders.items():
            pct = int(round(dsp.get(k, 0.0) * 100))
            sl.blockSignals(True)
            sl.setValue(pct)
            sl.blockSignals(False)
            if k in self.feature_val_labels:
                self.feature_val_labels[k].setText(f"{pct} %")

        self._updating_ui = False
        self._last_manual_eq_time = time.time()
        self.mode_label.setText(f"Genre EQ uygulandı: {profile}")

    def _save_preset(self) -> None:
        preset = self.preset_combo.currentText().strip()
        if not preset:
            return
        gains = tuple(s.value() / 10.0 for s in self.sliders)
        dsp = {k: sl.value() / 100.0 for k, sl in self.feature_sliders.items()}
        self.engine.save_preset(preset, gains, dsp)
        self.mode_label.setText(f"✓ Preset (EQ + DSP) kaydedildi: {preset}")
        if hasattr(self, "save_preset_btn") and self.save_preset_btn is not None:
            self.save_preset_btn.setText("✓ Kaydedildi")
            QTimer.singleShot(1200, lambda: self.save_preset_btn.setText("💾 Kaydet"))

    def _save_genre_profile(self) -> None:
        genre = self.genre_combo.currentText().strip()
        if not genre:
            return
        # Combo'da henüz yoksa ekle ve seçili yap
        if self.genre_combo.findText(genre) == -1:
            self.genre_combo.blockSignals(True)
            self.genre_combo.addItem(genre)
            self.genre_combo.setCurrentText(genre)
            self.genre_combo.blockSignals(False)

        gains = tuple(s.value() / 10.0 for s in self.sliders)
        dsp = {k: sl.value() / 100.0 for k, sl in self.feature_sliders.items()}
        self.engine.save_genre_profile(genre, gains, dsp)
        self.mode_label.setText(f"✓ Genre EQ (EQ + DSP) kaydedildi: {genre}")
        if hasattr(self, "save_genre_btn") and self.save_genre_btn is not None:
            self.save_genre_btn.setText("✓ Kaydedildi")
            QTimer.singleShot(1200, lambda: self.save_genre_btn.setText("💾 Kaydet"))

    def _gain_changed(self, idx: int, val: float) -> None:
        sign = "+" if val >= 0 else ""
        self.value_labels[idx].setText(f"{sign}{val:.1f} dB")
        if not self._updating_ui:
            self.engine.set_gain(idx, val)
            self._last_manual_eq_time = time.time()

    def _reset_eq(self) -> None:
        for sl in self.sliders:
            sl.setValue(0)
        self.mode_label.setText("EQ sıfırlandı")

    def _toggle_engine(self, checked: bool) -> None:
        self.engine.set_enabled(checked)
        self.toggle.setText("I" if checked else "O")
        self.mode_label.setText("DSP işlemi aktif" if checked else "Bypass — DSP devre dışı")

    def _volume_changed(self, value: int) -> None:
        vol = max(0.0, min(1.5, value / 100.0))
        self.vol_label.setText(f"{value} %")
        try:
            # System volume stays 0-1 range, engine gets full range
            sys_vol = min(1.0, vol)
            self.audio_manager.set_master(sys_vol, self.mute_button.isChecked())
            self.engine.set_volume(vol)
            self._last_system_volume = (round(sys_vol, 3), self.mute_button.isChecked())
        except (OSError, RuntimeError) as e:
            self.mode_label.setText(f"Ses sync hatası: {e}")

    def _balance_changed(self, value: int) -> None:
        self.engine.set_balance(value / 100.0)
        if value == 0:
            self.bal_label.setText("Merkez")
        elif value < 0:
            self.bal_label.setText(f"Sol {abs(value)} %")
        else:
            self.bal_label.setText(f"Sağ {value} %")

    def _mute_changed(self, muted: bool) -> None:
        self.mute_button.setText("Sessiz ✕" if muted else "Sustur")
        try:
            vol, _ = self.audio_manager.current_master()
            self.audio_manager.set_master(vol, muted)
            self.engine.set_volume(0.0 if muted else vol)
            self._last_system_volume = (round(vol, 3), muted)
        except (OSError, RuntimeError) as e:
            self.mode_label.setText(f"Sustur sync hatası: {e}")

    def _open_settings(self) -> None:
        if self._settings_dialog is None or not self._settings_dialog.isVisible():
            self._settings_dialog = SettingsDialog(self.engine, self.audio_manager, self)
            self._settings_dialog.show()
        else:
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _change_theme(self, name: str) -> None:
        if name not in THEMES:
            return
        self._current_theme = name
        _apply_accent(name)
        _save_theme(name)
        self._apply_theme()
        # Update spectrum canvas accent
        self.canvas.refresh_accent()
        # Regenerate tray icon with new accent
        try:
            self.tray.setIcon(create_auralab_icon(ACCENT))
            self.setWindowIcon(create_auralab_icon(ACCENT))
        except Exception:
            pass
        if hasattr(self, "_settings_dialog") and self._settings_dialog is not None and self._settings_dialog.isVisible():
            self._settings_dialog._apply_dialog_theme()
            self._settings_dialog._refresh_dialog_theme_swatches()

    # ── Timers & Refresh ──────────────────────────────────────────────────────
    def _blink_dot(self) -> None:
        self._dot_state = not self._dot_state
        dot = "⬤" if self._dot_state else "○"
        self.live_indicator.setText(f"{dot}  LIVE")

    def _refresh(self) -> None:
        snap = self.engine.snapshot()
        sr = snap.sample_rate
        lat = snap.latency_ms
        self.tech_label.setText(f"{sr:.0f} Hz  •  {lat:.1f} ms")
        if snap.error:
            self.mode_label.setText(f"⚠  {snap.error}")

    def _refresh_spectrum(self) -> None:
        """Dedicated fast spectrum update — uses lightweight get_spectrum_fast(), no full snapshot.
        update_spectrum() sadece animasyon durumunu ilerletir; canvas.update() buradan tetiklenir."""
        spectrum, gains, enabled = self.engine.get_spectrum_fast()
        self.canvas.update_spectrum(spectrum, gains, enabled)
        self.canvas.update()  # Tek yerden tetikle — update_spectrum içinde self.update() yok

    def _refresh_system_audio(self) -> None:
        try:
            vol, muted = self.audio_manager.current_master()
            cur = (round(vol, 3), muted)
            if cur != self._last_system_volume:
                self._last_system_volume = cur
                self.volume_slider.blockSignals(True)
                self.volume_slider.setValue(round(vol * 100))
                self.volume_slider.blockSignals(False)
                self.vol_label.setText(f"{round(vol * 100)} %")
                self.mute_button.blockSignals(True)
                self.mute_button.setChecked(muted)
                self.mute_button.blockSignals(False)
                self.mute_button.setText("Sessiz ✕" if muted else "Sustur")
                self.engine.set_volume(0.0 if muted else vol)
                self.audio_manager.set_master(vol, muted)

            # Windows varsayılan çıkış durumuna göre buton görünümünü senkronize et
            is_cable = self.audio_manager.is_cable_default()
            cur_direct = self.toggle.property("directSpeaker") == "true"
            if (not is_cable) != cur_direct:
                self._update_routing_button_state(is_cable)
        except Exception:
            pass

    def _refresh_spotify(self) -> None:
        track = self.spotify.current_track()
        if track is None or track.track_id == self._last_spotify_track_id:
            return
        self._last_spotify_track_id = track.track_id
        genre_text = track.genre or "bilinmiyor"
        self.track_label.setText(f"♫  {track.title}  —  {track.artist}")
        self.genre_detect_label.setText(f"Algılanan tür:  {genre_text}")

        # Türü sisteme ekle / eşle
        if track.genre:
            genre = self.engine.register_or_get_genre(track.genre)
            # Combo'da henüz yoksa ekle
            if self.genre_combo.findText(genre) == -1:
                self.genre_combo.blockSignals(True)
                self.genre_combo.addItem(genre)
                self.genre_combo.blockSignals(False)
        else:
            genre = "Flat"

        elapsed = time.time() - self._last_manual_eq_time
        if elapsed < self._MANUAL_EQ_GRACE:
            remaining = int(self._MANUAL_EQ_GRACE - elapsed)
            self.mode_label.setText(f"Spotify türü: {genre} — manuel EQ ({remaining}s sonra otomatik)")
            return

        self.genre_combo.blockSignals(True)
        self.genre_combo.setCurrentText(genre)
        self.genre_combo.blockSignals(False)
        self._genre_profile_changed(genre)
        self.mode_label.setText(f"Genre profili uygulandı: {genre}")
        self._current_spotify_genre = genre

    # ── Close Event & Clean Audio Restoration ─────────────────────────────────
    def closeEvent(self, event: Any) -> None:
        """Pencere kapatıldığında doğrudan tam kapan ve Windows hoparlörünü kesinlikle geri al!"""
        self.timer.stop()
        self.spectrum_timer.stop()
        self.system_timer.stop()
        self.spotify_timer.stop()
        self._dot_timer.stop()
        self.engine.close()
        self.tray.hide()
        try:
            self.audio_manager.restore()
        except Exception as err:
            pass
        event.accept()

    # ── Theme ─────────────────────────────────────────────────────────────────
    def _apply_theme(self) -> None:
        self.setStyleSheet(f"""
            /* === BASE === */
            QWidget {{ color:{TEXT}; font-family:'Segoe UI', sans-serif; font-size:12px; background:transparent; }}
            #root {{ background-color:{BG}; }}
            #scrollContent {{ background:transparent; }}
            #mainScroll {{ background:transparent; }}
            #footerWrap {{ background-color:{BG}; border-top:1px solid {BORDER}; }}
            QScrollBar:vertical {{
                background:{BG2}; width:8px; border:none; border-radius:4px;
            }}
            QScrollBar::handle:vertical {{
                background:{BORDER_HV}; min-height:30px; border-radius:4px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}

            /* === HEADER === */
            #logoLabel {{
                color:{TEXT}; font-size:20px; font-weight:900;
                letter-spacing:4px;
            }}
            #verBadge {{
                color:{ACCENT}; background:{ACCENT_BG};
                border:1px solid {ACCENT_DIM}; border-radius:4px;
                padding:2px 6px; font-size:10px; font-weight:700;
            }}
            #liveIndicator {{
                color:{ACCENT}; font-weight:700; font-size:11px;
                background:{ACCENT_BG}; border:1px solid {ACCENT_DIM};
                border-radius:12px; padding:3px 10px;
            }}
            #headerButton {{
                background:{BG3}; border:1px solid {BORDER}; border-radius:6px;
                color:{TEXT2}; padding:6px 14px; font-size:12px; font-weight:600;
            }}
            #headerButton:hover {{ border-color:{ACCENT}; color:{TEXT}; }}

            /* === HERO BAR === */
            #heroBar {{
                background:{BG2}; border:1px solid {BORDER};
                border-radius:10px;
            }}
            #trackLabel {{
                color:{TEXT}; font-size:13px; font-weight:700;
            }}
            #genreDetect {{
                color:{ACCENT}; font-size:11px; font-weight:600;
            }}
            #modeLabel {{
                color:{TEXT2}; font-size:11px;
            }}
            #techPill {{
                color:{TEXT3}; font-size:11px; font-family:'Consolas', monospace;
                background:{BG3}; border:1px solid {BORDER}; border-radius:6px;
                padding:4px 10px;
            }}
            #powerToggle {{
                background:{ACCENT_BG}; border:2px solid {ACCENT};
                border-radius:22px; color:{ACCENT}; font-size:20px; font-weight:900;
            }}
            #powerToggle:hover {{ background:{ACCENT}; color:{BG}; }}
            #powerToggle[directSpeaker="true"] {{
                background:#241608; border:2px solid {WARN}; color:{WARN};
            }}
            #powerToggle[directSpeaker="true"]:hover {{
                background:{WARN}; color:{BG};
            }}

            /* === CARDS === */
            #card {{
                background:{BG2}; border:1px solid {BORDER};
                border-radius:10px;
            }}
            #cardTitle {{
                color:{TEXT2}; font-size:11px; font-weight:700;
                letter-spacing:1.5px;
            }}
            #spectrumTitle {{
                color:{ACCENT}; font-size:11px; font-weight:800;
                letter-spacing:2px;
            }}
            #spectrumRange {{
                color:{TEXT3}; font-size:10px; font-weight:600;
                font-family:'Consolas', monospace;
            }}
            #cardSep {{
                border:none; border-top:1px solid {BORDER}; max-height:1px;
            }}

            /* === EQUALIZER === */
            #freqLabel {{ color:{TEXT3}; font-size:10px; font-weight:600; }}
            #bandLabel {{ color:{TEXT}; font-size:11px; font-weight:700; }}
            #gainValue {{
                color:{ACCENT}; font-size:11px; font-weight:700;
                font-family:'Consolas', monospace; min-width:55px;
            }}
            #eqSlider::groove:vertical {{
                width:4px; background:{BORDER}; border-radius:2px;
            }}
            #eqSlider::add-page:vertical {{
                background:{ACCENT}; border-radius:2px;
            }}
            #eqSlider::sub-page:vertical {{
                background:{BORDER}; border-radius:2px;
            }}
            #eqSlider::handle:vertical {{
                background:#fff; border:2px solid {ACCENT};
                height:16px; margin:0 -6px; border-radius:8px;
            }}
            #eqSlider::handle:vertical:hover {{
                background:{ACCENT}; border-color:#fff;
            }}

            /* === BUTTONS & COMBOS === */
            #mainCombo {{
                background:{BG3}; border:1px solid {BORDER};
                border-radius:6px; color:{TEXT}; padding:5px 10px;
                font-size:11px; min-width:110px;
            }}
            #mainCombo:hover {{ border-color:{BORDER_HV}; }}
            #mainCombo:disabled {{
                background:{BG}; border:1px solid {BORDER};
                color:{TEXT3}; border-radius:6px; padding:5px 10px; font-size:11px;
            }}
            QLabel:disabled {{ color:{TEXT3}; }}
            #mainCombo QAbstractItemView {{
                background:{BG2}; color:{TEXT}; selection-background-color:{BG3};
                border:1px solid {BORDER};
            }}
            #compactSaveBtn {{
                background:{ACCENT}; color:{BG}; border:none;
                border-radius:6px; padding:4px 10px; font-weight:700; font-size:11px;
                min-height:20px;
            }}
            #compactSaveBtn:hover {{ background:{ACCENT}; opacity:0.9; }}
            #accentButton {{
                background:{ACCENT}; color:{BG}; border:none;
                border-radius:6px; padding:5px 12px; font-weight:700; font-size:11px;
            }}
            #accentButton:hover {{ background:{ACCENT}; }}
            #ghostButton {{
                background:transparent; border:1px solid {BORDER};
                border-radius:6px; color:{TEXT2}; padding:5px 10px; font-size:11px;
            }}
            #ghostButton:hover {{ border-color:{BORDER_HV}; color:{TEXT}; }}
            #pillButton {{
                background:{BG3}; border:1px solid {BORDER};
                border-radius:6px; color:{TEXT2}; padding:5px 12px;
                font-size:11px; font-weight:600;
            }}
            #pillButton:checked {{
                background:{ACCENT_BG}; border-color:{ACCENT}; color:{ACCENT};
            }}
            #lrSeparateButton {{
                background:{BG3}; border:1.5px solid {BORDER_HV};
                border-radius:8px; color:{TEXT}; padding:6px 14px;
                font-size:12px; font-weight:700; min-height:20px;
            }}
            #lrSeparateButton:hover {{
                border-color:{ACCENT}; color:#fff; background:{BG2};
            }}
            #lrSeparateButton:checked {{
                background:{ACCENT_BG}; border:1.5px solid {ACCENT}; color:{ACCENT}; font-weight:800;
            }}

            /* === DSP SLIDERS === */
            #dspTitle {{ color:{TEXT}; font-weight:600; font-size:11px; }}
            #dspValue {{
                color:{ACCENT}; font-size:11px; font-weight:700;
                font-family:'Consolas', monospace;
            }}
            #dspSlider::groove:horizontal {{
                height:4px; background:{BORDER}; border-radius:2px;
            }}
            #dspSlider::sub-page:horizontal {{
                background:{ACCENT}; border-radius:2px;
            }}
            #dspSlider::handle:horizontal {{
                background:#fff; border:2px solid {ACCENT};
                width:12px; margin:-4px 0; border-radius:6px;
            }}

            /* === CALIBRATION SLIDERS === */
            #calLabel {{
                color:{ACCENT}; font-size:11px; font-weight:700;
                font-family:'Consolas', monospace; min-width:42px;
            }}
            #speakerSliderL::groove:horizontal, #speakerSliderR::groove:horizontal {{
                height:4px; background:{BORDER}; border-radius:2px;
            }}
            #speakerLabelL {{ color:{ACCENT}; font-weight:600; }}
            #speakerLabelR {{ color:{TEXT2}; font-weight:600; }}
            #speakerSliderL::sub-page:horizontal {{ background:{ACCENT}; border-radius:2px; }}
            #speakerSliderL::handle:horizontal {{
                background:#fff; border:2px solid {ACCENT}; width:12px; margin:-4px 0; border-radius:6px;
            }}
            #speakerSliderR::sub-page:horizontal {{ background:{ACCENT_DIM}; border-radius:2px; }}
            #speakerSliderR::handle:horizontal {{
                background:#fff; border:2px solid {ACCENT_DIM}; width:12px; margin:-4px 0; border-radius:6px;
            }}

            /* === FOOTER === */
            #footerBar {{
                background:{BG2}; border:1px solid {BORDER};
                border-radius:10px;
            }}
            #footerLabel {{ color:{TEXT2}; font-size:11px; font-weight:600; }}
            #footerVal {{
                color:{TEXT}; font-size:11px; font-weight:700;
                font-family:'Consolas', monospace; min-width:40px;
            }}
            #footerSlider::groove:horizontal {{
                height:4px; background:{BORDER}; border-radius:2px;
            }}
            #footerSlider::sub-page:horizontal {{
                background:{ACCENT}; border-radius:2px;
            }}
            #footerSlider::handle:horizontal {{
                background:#fff; border:2px solid {ACCENT};
                width:12px; margin:-4px 0; border-radius:6px;
            }}
            #muteButton {{
                background:{BG3}; border:1px solid {BORDER};
                border-radius:6px; color:{TEXT2}; padding:4px 12px; font-size:11px;
            }}
            #muteButton:checked {{
                background:#2a0e0e; border-color:#d94040; color:#d94040; font-weight:700;
            }}
            #vSep {{ border:none; border-left:1px solid {BORDER}; max-width:1px; }}
        """)


# ──────────────────────────────────────────────────────────────────────────────
class ThreadTarget:
    def __init__(self, target: Any) -> None:
        import threading
        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()


# ──────────────────────────────────────────────────────────────────────────────
def run() -> int:
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("auralab.desktop.app.2.3")
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("AuraLab")

    # Set application icon
    icon = create_auralab_icon(ACCENT)
    app.setWindowIcon(icon)

    lock_path = os.path.join(os.environ.get("TEMP", "."), "auto_eq.lock")
    instance_lock = QLockFile(lock_path)
    instance_lock.setStaleLockTime(5000)
    if not instance_lock.tryLock(50):
        # Eğer kilit dosyası önceki çökmeden kalmış ve bayatlamışsa kurtarmayı dene
        if instance_lock.isLocked():
            QMessageBox.warning(None, "AuraLab", "AuraLab zaten çalışıyor.")
            return 1
        instance_lock.removeStaleLockFile()
        if not instance_lock.tryLock(50):
            QMessageBox.warning(None, "AuraLab", "AuraLab zaten çalışıyor.")
            return 1

    audio_manager = None
    try:
        audio_manager = SystemAudioManager()
        audio_manager.begin()
        engine = AudioEngine()
        window = AutoEQWindow(engine, audio_manager)
        window.setWindowIcon(icon)

        # Pencereyi ekranın ortasına taşı
        screen = app.primaryScreen()
        if screen is not None:
            screen_geom = screen.availableGeometry()
            win_geom = window.frameGeometry()
            center = screen_geom.center()
            win_geom.moveCenter(center)
            window.move(win_geom.topLeft())

        window.show()

        # Qt kapanırken hoparlörü mutlaka geri yükle
        app.aboutToQuit.connect(audio_manager.restore)

        def _sigint_handler(signum: int, frame: Any) -> None:
            if audio_manager is not None:
                audio_manager.restore()
            app.quit()

        signal.signal(signal.SIGINT, _sigint_handler)

        # Qt event loop'unu SIGINT için düzenli uyandır
        _sigint_poll = QTimer()
        _sigint_poll.start(200)
        _sigint_poll.timeout.connect(lambda: None)

        return app.exec()
    except Exception as err:
        traceback.print_exc()
        try:
            with open(Path.cwd() / "crash.log", "a", encoding="utf-8") as f:
                f.write(f"\n--- STARTUP CRASH AT {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                f.write(traceback.format_exc())
        except Exception:
            pass
        if audio_manager is not None:
            try:
                audio_manager.restore()
            except Exception:
                pass
        QMessageBox.critical(None, "AuraLab", f"Başlatılamadı:\n{err}")
        return 1
    finally:
        if audio_manager is not None:
            try:
                audio_manager.restore()
            except Exception:
                pass
        instance_lock.unlock()