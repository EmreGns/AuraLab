"""Programmatic AuraLab application icon — generated at runtime via QPainter."""

from __future__ import annotations

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import (
    QColor,
    QIcon,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QRadialGradient,
)


def create_auralab_icon(accent: str = "#8df230", size: int = 128) -> QIcon:
    """Create a premium AuraLab icon with a waveform motif."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    cx, cy = size // 2, size // 2
    r = size // 2 - 2

    # Background circle — dark gradient
    bg_grad = QRadialGradient(cx, cy, r)
    bg_grad.setColorAt(0.0, QColor("#141e2a"))
    bg_grad.setColorAt(1.0, QColor("#070b0f"))
    p.setBrush(bg_grad)
    p.setPen(QPen(QColor(accent), 2.0))
    p.drawEllipse(QRect(2, 2, size - 4, size - 4))

    # Waveform — three bars with accent gradient
    accent_color = QColor(accent)
    bar_grad = QLinearGradient(0, cy - r * 0.4, 0, cy + r * 0.4)
    bar_grad.setColorAt(0.0, accent_color)
    bar_grad.setColorAt(0.5, accent_color.lighter(130))
    bar_grad.setColorAt(1.0, accent_color)

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(bar_grad)

    bar_w = max(4, int(size * 0.09))
    gap = max(6, int(size * 0.12))
    heights = [0.55, 0.35, 0.45]

    total_w = len(heights) * bar_w + (len(heights) - 1) * gap
    start_x = cx - total_w // 2

    for i, h_ratio in enumerate(heights):
        bx = start_x + i * (bar_w + gap)
        bh = int(r * h_ratio * 2)
        by = cy - bh // 2
        p.drawRoundedRect(bx, by, bar_w, bh, bar_w // 2, bar_w // 2)

    # Subtle glow ring
    glow_pen = QPen(QColor(accent))
    glow_pen.setWidth(1)
    p.setPen(glow_pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setOpacity(0.3)
    p.drawEllipse(QRect(6, 6, size - 12, size - 12))

    p.end()
    return QIcon(pixmap)
