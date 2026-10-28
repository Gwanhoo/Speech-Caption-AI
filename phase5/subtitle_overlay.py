"""Separate always-on-top window for the existing subtitle presentation snapshot."""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, Qt, Slot
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QMouseEvent
from PySide6.QtWidgets import QFrame, QLabel, QLayout, QSizePolicy, QVBoxLayout, QWidget

try:
    from .subtitle_presentation import SubtitleFeedEntry, trace_display
except ImportError:  # Direct execution: python phase5/gui_app.py
    from subtitle_presentation import SubtitleFeedEntry, trace_display  # type: ignore[no-redef]


@dataclass(frozen=True)
class OverlayStyle:
    font_size: int = 28
    background_opacity: int = 72
    bold: bool = True
    text_color: str = "#FFFFFF"


@dataclass(frozen=True)
class OverlayLayoutMetrics:
    source_rows: int
    visible_rows: int
    required_height: int
    maximum_height: int
    window_height: int


class SubtitleOverlay(QWidget):
    """Draggable, transparent overlay that renders an existing UI snapshot."""

    MAX_ROWS = 3
    HORIZONTAL_MARGIN = 20
    VERTICAL_MARGIN = 8
    ROW_SPACING = 2
    FRAME_BORDER_HEIGHT = 2
    EMPTY_HEIGHT = 1
    MAX_SCREEN_HEIGHT_RATIO = 0.30

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._style = OverlayStyle()
        self._drag_offset: QPoint | None = None
        self._was_dragged = False
        self._keep_empty_visible = False
        self._entries: tuple[SubtitleFeedEntry, ...] = ()
        self._source_entries: tuple[SubtitleFeedEntry, ...] = ()
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
        # Without this, Qt caches the previous child minimum and a
        # 3 -> 2 -> 1 -> 0 update shrinks one event late.
        root.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self._card = QFrame(self)
        self._card.setObjectName("overlayCard")
        root.addWidget(self._card)
        layout = QVBoxLayout(self._card)
        layout.setContentsMargins(
            self.HORIZONTAL_MARGIN,
            self.VERTICAL_MARGIN,
            self.HORIZONTAL_MARGIN,
            self.VERTICAL_MARGIN,
        )
        layout.setSpacing(self.ROW_SPACING)
        self._labels: list[QLabel] = []
        for _ in range(self.MAX_ROWS):
            label = QLabel(self._card)
            label.setAlignment(
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
            )
            label.setWordWrap(True)
            label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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

    @property
    def layout_metrics(self) -> OverlayLayoutMetrics:
        return OverlayLayoutMetrics(
            source_rows=len(self._source_entries),
            visible_rows=len(self._entries),
            required_height=self._required_height(self._entries),
            maximum_height=self._maximum_height(),
            window_height=self.height(),
        )

    def _place_on_primary_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(900, self.EMPTY_HEIGHT)
            return
        available = screen.availableGeometry()
        width = min(max(320, round(available.width() * 0.82)), available.width())
        height = self.EMPTY_HEIGHT
        x = available.x() + (available.width() - width) // 2
        bottom_margin = max(16, round(available.height() * 0.04))
        y = available.y() + available.height() - height - bottom_margin
        self.setGeometry(x, y, width, height)

    @Slot(object)
    def set_entries(self, entries: object) -> None:
        snapshot = tuple(entries) if isinstance(entries, (tuple, list)) else ()
        self._source_entries = tuple(
            entry for entry in snapshot[-self.MAX_ROWS:] if isinstance(entry, SubtitleFeedEntry)
        )
        self._entries = self._fit_entries_to_screen(self._source_entries)
        for index, label in enumerate(self._labels):
            if index < len(self._entries):
                entry = self._entries[index]
                label.setText(entry.text)
                label.setVisible(True)
                self._apply_label_style(label, entry.status)
            else:
                label.clear()
                label.setToolTip("")
                label.setVisible(False)
        self._resize_for_entries()
        trace_display("overlay_delivery", entries=self._entries, visible=self.isVisible(),
                      source_rows=len(self._source_entries))

    def clear_entries(self) -> None:
        self.set_entries(())

    def set_keep_empty_visible(self, active: bool) -> None:
        """Keep a blank, styled bar while the GUI caption session is active."""
        self._keep_empty_visible = active
        self._resize_for_entries()

    def apply_style(self, style: OverlayStyle) -> None:
        color = QColor(style.text_color)
        self._style = OverlayStyle(
            font_size=max(16, min(52, style.font_size)),
            background_opacity=max(20, min(95, style.background_opacity)),
            bold=style.bold,
            text_color=color.name().upper() if color.isValid() else "#FFFFFF",
        )
        self._entries = self._fit_entries_to_screen(self._source_entries)
        for index, label in enumerate(self._labels):
            if index < len(self._entries):
                entry = self._entries[index]
                label.setText(entry.text)
                label.setVisible(True)
            else:
                label.clear()
                label.setToolTip("")
                label.setVisible(False)
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
        self._resize_for_entries()

    def _apply_label_style(self, label: QLabel, status: str) -> None:
        font = QFont()
        font.setPointSize(self._style.font_size)
        font.setBold(self._style.bold)
        label.setFont(font)
        # PARTIAL is intentionally subtle: no status text or speaker label.
        text_alpha = 255 if status == "final" else 210
        color = QColor(self._style.text_color)
        label.setStyleSheet(
            "color: rgba("
            f"{color.red()}, {color.green()}, {color.blue()}, {text_alpha}"
            ");"
        )

    def _resize_for_entries(self) -> None:
        """Match the window to its rows and cap extreme content to a compact bar."""
        if not self._entries:
            self._card.setVisible(self._keep_empty_visible)
            self._card.setToolTip("")
            self._set_window_height(self._required_height(()))
            return
        self._card.show()
        desired_heights = [
            self._entry_height(entry, self._text_width()) for entry in self._entries
        ]
        non_text_height = self._non_text_height(len(self._entries))
        maximum_height = self._maximum_height()
        height = min(maximum_height, non_text_height + sum(desired_heights))
        available_text_height = max(1, height - non_text_height)
        card_tooltip = ""
        for index, desired_height in enumerate(desired_heights):
            remaining_rows = len(desired_heights) - index - 1
            row_height = min(desired_height, max(1, available_text_height - remaining_rows))
            label = self._labels[index]
            display_text = self._display_text_for_height(
                self._entries[index], self._text_width(), row_height
            )
            label.setText(display_text)
            label.setToolTip(
                self._entries[index].text
                if display_text != self._entries[index].text
                else ""
            )
            if display_text != self._entries[index].text:
                card_tooltip = self._entries[index].text
            label.setFixedHeight(row_height)
            available_text_height -= row_height
        self._card.setToolTip(card_tooltip)
        self._set_window_height(height)

    def _set_window_height(self, height: int) -> None:
        height_changed = height != self.height()
        # On Windows, an unconstrained native tool window can retain the OS
        # minimum tracking height (118px here) when a row needs only 67px.
        # This frameless bar already owns its height; enforce that computed
        # height for both shrink and growth, including the empty 1px state.
        self.setFixedHeight(height)
        if not height_changed:
            return
        if self._was_dragged:
            self.resize(self.width(), height)
            return
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(self.width(), height)
            return
        available = screen.availableGeometry()
        bottom_margin = max(16, round(available.height() * 0.04))
        self.setGeometry(
            self.x(),
            available.y() + available.height() - height - bottom_margin,
            self.width(),
            height,
        )

    def _fit_entries_to_screen(
        self, candidates: tuple[SubtitleFeedEntry, ...]
    ) -> tuple[SubtitleFeedEntry, ...]:
        """Prefer the newest readable entries when several wrapped rows do not fit."""
        selected: list[SubtitleFeedEntry] = []
        maximum_height = self._maximum_height()
        for entry in reversed(candidates):
            proposed = [entry, *selected]
            required_height = self._required_height(tuple(proposed))
            if required_height > maximum_height and selected:
                break
            selected = proposed
        return tuple(selected)

    def _required_height(self, entries: tuple[SubtitleFeedEntry, ...]) -> int:
        if not entries:
            if self._keep_empty_visible:
                return self._non_text_height(1) + self._font_line_height()
            return self.EMPTY_HEIGHT
        return self._non_text_height(len(entries)) + sum(
            self._entry_height(entry, self._text_width()) for entry in entries
        )

    def _non_text_height(self, row_count: int) -> int:
        spacing = self.ROW_SPACING * max(0, row_count - 1)
        return self.VERTICAL_MARGIN * 2 + self.FRAME_BORDER_HEIGHT + spacing

    def _text_width(self) -> int:
        return max(1, self.width() - self.HORIZONTAL_MARGIN * 2 - 2)

    def _maximum_height(self) -> int:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return max(180, self._non_text_height(1) + self._font_line_height())
        available_height = screen.availableGeometry().height()
        one_line_height = self._non_text_height(1) + self._font_line_height()
        return max(one_line_height, round(available_height * self.MAX_SCREEN_HEIGHT_RATIO))

    def _font_line_height(self) -> int:
        font = QFont()
        font.setPointSize(self._style.font_size)
        font.setBold(self._style.bold)
        return QFontMetrics(font).height()

    def _entry_height(self, entry: SubtitleFeedEntry, width: int) -> int:
        font = QFont()
        font.setPointSize(self._style.font_size)
        font.setBold(self._style.bold)
        metrics = QFontMetrics(font)
        bounds = metrics.boundingRect(
            0,
            0,
            width,
            10000,
            int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignHCenter),
            entry.text,
        )
        return max(metrics.height(), bounds.height())

    def _display_text_for_height(
        self, entry: SubtitleFeedEntry, width: int, maximum_height: int
    ) -> str:
        """Return a whole-line display string when one entry exceeds the height cap."""
        if self._entry_height(entry, width) <= maximum_height:
            return entry.text

        best_length = 0
        low, high = 1, len(entry.text)
        while low <= high:
            length = (low + high) // 2
            candidate = entry.text[:length].rstrip() + "…"
            candidate_entry = SubtitleFeedEntry(
                speaker_id=entry.speaker_id,
                utterance_id=entry.utterance_id,
                sequence=entry.sequence,
                status=entry.status,
                text=candidate,
            )
            if self._entry_height(candidate_entry, width) <= maximum_height:
                best_length = length
                low = length + 1
            else:
                high = length - 1

        return entry.text[:best_length].rstrip() + "…"

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton and self._drag_offset is not None:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            self._was_dragged = True
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._drag_offset = None
        super().mouseReleaseEvent(event)
