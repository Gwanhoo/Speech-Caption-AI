"""Separate always-on-top window for the existing subtitle presentation snapshot."""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, Qt, Slot
from PySide6.QtGui import QFont, QGuiApplication, QMouseEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

try:
    from .subtitle_presentation import SubtitleFeedEntry
except ImportError:  # Direct execution: python phase5/gui_app.py
    from subtitle_presentation import SubtitleFeedEntry  # type: ignore[no-redef]


@dataclass(frozen=True)
class OverlayStyle:
    font_size: int = 28
    background_opacity: int = 72
    bold: bool = True


class SubtitleOverlay(QWidget):
    """Draggable, transparent overlay that renders an existing UI snapshot."""

    MAX_ROWS = 3

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._style = OverlayStyle()
        self._drag_offset: QPoint | None = None
        self._entries: tuple[SubtitleFeedEntry, ...] = ()
        self.setWindowTitle("AI 실시간 자막 Overlay")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._card = QFrame(self)
        self._card.setObjectName("overlayCard")
        root.addWidget(self._card)
        layout = QVBoxLayout(self._card)
        layout.setContentsMargins(28, 18, 28, 18)
        layout.setSpacing(8)
        self._labels: list[QLabel] = []
        for _ in range(self.MAX_ROWS):
            label = QLabel(self._card)
            label.setAlignment(
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
            )
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            layout.addWidget(label)
            self._labels.append(label)

        self._place_on_primary_screen()
        self.apply_style(self._style)
        self.clear_entries()

    @property
    def style_config(self) -> OverlayStyle:
        return self._style

    @property
    def entries(self) -> tuple[SubtitleFeedEntry, ...]:
        return self._entries

    def _place_on_primary_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(900, 210)
            return
        available = screen.availableGeometry()
        width = min(max(320, round(available.width() * 0.82)), available.width())
        height = min(max(150, round(available.height() * 0.20)), 230)
        x = available.x() + (available.width() - width) // 2
        bottom_margin = max(16, round(available.height() * 0.04))
        y = available.y() + available.height() - height - bottom_margin
        self.setGeometry(x, y, width, height)

    @Slot(object)
    def set_entries(self, entries: object) -> None:
        snapshot = tuple(entries) if isinstance(entries, (tuple, list)) else ()
        self._entries = tuple(
            entry for entry in snapshot[-self.MAX_ROWS:] if isinstance(entry, SubtitleFeedEntry)
        )
        for index, label in enumerate(self._labels):
            if index < len(self._entries):
                entry = self._entries[index]
                label.setText(entry.text)
                label.setVisible(True)
                self._apply_label_style(label, entry.status)
            else:
                label.clear()
                label.setVisible(False)

    def clear_entries(self) -> None:
        self.set_entries(())

    def apply_style(self, style: OverlayStyle) -> None:
        self._style = OverlayStyle(
            font_size=max(16, min(52, style.font_size)),
            background_opacity=max(20, min(95, style.background_opacity)),
            bold=style.bold,
        )
        alpha = round(255 * self._style.background_opacity / 100)
        self._card.setStyleSheet(
            "QFrame#overlayCard {"
            f"background-color: rgba(10, 15, 25, {alpha});"
            "border: 1px solid rgba(255, 255, 255, 72);"
            "border-radius: 18px;"
            "}"
        )
        for index, entry in enumerate(self._entries):
            self._apply_label_style(self._labels[index], entry.status)

    def _apply_label_style(self, label: QLabel, status: str) -> None:
        font = QFont()
        font.setPointSize(self._style.font_size)
        font.setBold(self._style.bold if status == "final" else False)
        label.setFont(font)
        # PARTIAL is intentionally subtle: no status text or speaker label.
        text_alpha = 255 if status == "final" else 210
        label.setStyleSheet(f"color: rgba(255, 255, 255, {text_alpha});")

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton and self._drag_offset is not None:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._drag_offset = None
        super().mouseReleaseEvent(event)
