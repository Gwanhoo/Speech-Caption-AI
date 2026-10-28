from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

# See live_caption_controller.py: this import must precede PySide6 on Python 3.10.
from typing_extensions import Self as _TypingExtensionsSelf  # noqa: F401

from PySide6.QtCore import QAbstractListModel, QModelIndex, QSize, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QColorDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStyledItemDelegate,
    QSlider,
    QSpinBox,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

try:
    from .live_caption_controller import (
        DEFAULT_REMOTE_SERVER_URL,
        LOGGER,
        LiveCaptionConfig,
        LiveCaptionController,
        SpeakerMode,
        configure_gui_runtime,
        shutdown_gui_runtime,
    )
    from .subtitle_presentation import SubtitleFeedEntry, SubtitlePresentationState, trace_display
    from .subtitle_overlay import OverlayStyle, SubtitleOverlay
    from audio_devices import list_output_devices
except ImportError:  # Direct execution: python phase5/gui_app.py
    from live_caption_controller import (  # type: ignore[no-redef]
        DEFAULT_REMOTE_SERVER_URL,
        LOGGER,
        LiveCaptionConfig,
        LiveCaptionController,
        SpeakerMode,
        configure_gui_runtime,
        shutdown_gui_runtime,
    )
    from subtitle_presentation import SubtitleFeedEntry, SubtitlePresentationState, trace_display  # type: ignore[no-redef]
    from subtitle_overlay import OverlayStyle, SubtitleOverlay  # type: ignore[no-redef]
    from audio_devices import list_output_devices  # type: ignore[no-redef]


APP_STYLE = """
QMainWindow, QWidget#centralWidget { background: #f4f7fb; color: #172033; }
QLabel#title { color: #111827; font-size: 30px; font-weight: 700; }
QLabel#hint { color: #64748b; font-size: 12px; }
QFrame#infoCard, QFrame#subtitleCard, QFrame#statusCard {
    background: white;
    border: 1px solid #dbe3ef;
    border-radius: 12px;
}
QFrame#overlaySettingsCard { background: #ffffff; border: 1px solid #dbe3ef; border-radius: 12px; }
QLabel#fieldTitle { color: #526078; font-weight: 600; }
QLineEdit {
    background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 7px;
    padding: 8px 10px; color: #1e293b;
}
QListView#subtitleFeed {
    background: #fbfdff; border: 0; padding: 20px 28px; color: #0f172a;
    outline: 0;
}
QPushButton { border-radius: 8px; padding: 11px 22px; font-size: 15px; font-weight: 700; }
QPushButton#startButton { background: #2563eb; color: white; border: 1px solid #1d4ed8; }
QPushButton#startButton:hover { background: #1d4ed8; }
QPushButton#stopButton { background: #fff1f2; color: #be123c; border: 1px solid #fecdd3; }
QPushButton#stopButton:hover { background: #ffe4e6; }
QPushButton:disabled { background: #e2e8f0; color: #94a3b8; border-color: #e2e8f0; }
QLabel#statusText { font-size: 14px; font-weight: 600; }
QLabel#latencyText { color: #475569; font-size: 13px; }
"""


class SubtitleFeedModel(QAbstractListModel):
    """Qt adapter for the bounded reader-focused subtitle presentation state."""

    entry_upserted = Signal(int, bool)
    entries_changed = Signal(object)
    MAX_FINAL_ENTRIES = SubtitlePresentationState.MAX_FINAL_ENTRIES
    MAX_ENTRIES = MAX_FINAL_ENTRIES + 1
    EXPIRY_POLL_MS = 100
    SpeakerRole = Qt.ItemDataRole.UserRole + 1
    UtteranceRole = Qt.ItemDataRole.UserRole + 2
    SequenceRole = Qt.ItemDataRole.UserRole + 3
    StatusRole = Qt.ItemDataRole.UserRole + 4

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: list[SubtitleFeedEntry] = []
        self._presentation = SubtitlePresentationState()
        self._expiry_timer = QTimer(self)
        self._expiry_timer.setInterval(self.EXPIRY_POLL_MS)
        self._expiry_timer.timeout.connect(self.expire_final_entries)
        self._expiry_timer.start()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._entries)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not 0 <= index.row() < len(self._entries):
            return None
        entry = self._entries[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return entry.text
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        if role == self.SpeakerRole:
            return entry.speaker_id
        if role == self.UtteranceRole:
            return entry.utterance_id
        if role == self.SequenceRole:
            return entry.sequence
        if role == self.StatusRole:
            return entry.status
        return None

    def clear(self) -> None:
        self.beginResetModel()
        self._presentation.clear()
        self._entries = []
        self.endResetModel()
        self.entries_changed.emit(())

    def upsert_event(self, event: dict[str, Any]) -> bool:
        if not self._presentation.apply(event):
            return False
        self._refresh_snapshot()
        self.entry_upserted.emit(len(self._entries) - 1, True)
        return True

    @Slot()
    def expire_final_entries(self) -> None:
        if self._presentation.expire():
            self._refresh_snapshot()

    def _refresh_snapshot(self) -> None:
        self.beginResetModel()
        self._entries = list(self._presentation.entries)
        self.endResetModel()
        trace_display("gui_model_delivery", entries=self._entries)
        self.entries_changed.emit(tuple(self._entries))

    def entry_at(self, row: int) -> SubtitleFeedEntry:
        return self._entries[row]

    @property
    def entries(self) -> tuple[SubtitleFeedEntry, ...]:
        return tuple(self._entries)

class SubtitleFeedDelegate(QStyledItemDelegate):
    """Large, centered, wrapped text rendering kept separate for future overlay use."""

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:  # noqa: N802
        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        view = self.parent()
        viewport_width = (
            view.viewport().width() if isinstance(view, QListView) else option.rect.width()
        )
        width = max(320, viewport_width - 56)
        bounds = option.fontMetrics.boundingRect(
            0,
            0,
            width,
            1000,
            int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignHCenter),
            text,
        )
        return QSize(width, max(72, bounds.height() + 34))


class SubtitleFeedView(QListView):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("subtitleFeed")
        self.setAccessibleName("실시간 자막")
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setWordWrap(True)
        self.setSpacing(14)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        font = QFont()
        font.setPointSize(24)
        font.setWeight(QFont.Weight.DemiBold)
        self.setFont(font)
        self.setItemDelegate(SubtitleFeedDelegate(self))

    @Slot(int, bool)
    def keep_latest_visible(self, _row: int, _inserted: bool) -> None:
        self.scrollToBottom()


class MainWindow(QMainWindow):
    def __init__(
        self,
        controller: LiveCaptionController | None = None,
        parent: QWidget | None = None,
        *,
        config: LiveCaptionConfig | None = None,
    ) -> None:
        super().__init__(parent)
        # A closed top-level widget is hidden by default, leaving its active
        # child timers reachable until Python's cyclic GC runs.  GC may run on
        # a caption worker thread, where destroying GUI-thread Qt objects is
        # unsafe.  Let Qt delete the complete object tree on the GUI thread as
        # soon as an accepted close finishes.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.controller = controller or LiveCaptionController()
        self._launch_config = config or LiveCaptionConfig()
        self._close_when_finished = False
        self.overlay: SubtitleOverlay | None = None
        self.setWindowTitle("AI 실시간 자막")
        self.resize(1080, 820)

        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(15)

        title = QLabel("AI 실시간 자막")
        title.setObjectName("title")
        root.addWidget(title)
        hint = QLabel("Windows 시스템 오디오를 RunPod GPU에서 처리합니다.")
        hint.setObjectName("hint")
        root.addWidget(hint)

        info_card = QFrame()
        info_card.setObjectName("infoCard")
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(18, 15, 18, 15)
        info_layout.setSpacing(10)

        server_row = QHBoxLayout()
        server_title = QLabel("서버 상태")
        server_title.setObjectName("fieldTitle")
        server_title.setFixedWidth(95)
        self.server_state_label = QLabel("● 확인 전")
        self.server_state_label.setObjectName("serverState")
        self._set_indicator(self.server_state_label, "#94a3b8")
        server_row.addWidget(server_title)
        server_row.addWidget(self.server_state_label)
        server_row.addStretch(1)
        info_layout.addLayout(server_row)

        mode_row = QHBoxLayout()
        mode_title = QLabel("처리 모드")
        mode_title.setObjectName("fieldTitle")
        mode_title.setFixedWidth(95)
        self.speaker_mode_combo = QComboBox()
        self.speaker_mode_combo.setObjectName("speakerMode")
        self.speaker_mode_combo.addItem("일반 모드", "single")
        self.speaker_mode_combo.addItem("겹친 음성 모드", "overlap")
        overlap_item = self.speaker_mode_combo.model().item(1)
        if overlap_item is not None:
            overlap_item.setEnabled(False)
            overlap_item.setToolTip("기능 개발 중입니다.")
        self.speaker_mode_combo.setToolTip("겹친 음성 모드는 기능 개발 중입니다.")
        mode_row.addWidget(mode_title)
        mode_row.addWidget(self.speaker_mode_combo, 1)
        info_layout.addLayout(mode_row)

        url_row = QHBoxLayout()
        url_title = QLabel("서버 URL")
        url_title.setObjectName("fieldTitle")
        url_title.setFixedWidth(95)
        self.server_url_edit = QLineEdit(DEFAULT_REMOTE_SERVER_URL)
        self.server_url_edit.setObjectName("serverUrl")
        url_row.addWidget(url_title)
        url_row.addWidget(self.server_url_edit, 1)
        info_layout.addLayout(url_row)

        device_row = QHBoxLayout()
        device_title = QLabel("오디오 장치")
        device_title.setObjectName("fieldTitle")
        device_title.setFixedWidth(95)
        self.output_device_combo = QComboBox()
        self.output_device_combo.setObjectName("outputDevice")
        self.device_label = QLabel("자막 시작 시 WASAPI loopback 장치를 확인합니다.")
        self.device_label.setObjectName("deviceState")
        self.device_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.refresh_device_button = QPushButton("새로고침")
        self.refresh_device_button.setObjectName("refreshDeviceButton")
        self.refresh_device_button.setMinimumWidth(92)
        self._has_available_output_devices = self._refresh_output_devices(
            announce_selection_change=False,
        )
        device_row.addWidget(device_title)
        device_row.addWidget(self.output_device_combo, 1)
        device_row.addWidget(self.refresh_device_button)
        device_row.addWidget(self.device_label, 1)
        info_layout.addLayout(device_row)
        root.addWidget(info_card)

        controls = QHBoxLayout()
        controls.addStretch(1)
        self.start_button = QPushButton("자막 시작")
        self.start_button.setObjectName("startButton")
        self.start_button.setMinimumWidth(145)
        self.stop_button = QPushButton("중지")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setMinimumWidth(110)
        self.stop_button.setEnabled(False)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        root.addLayout(controls)

        overlay_card = QFrame()
        overlay_card.setObjectName("overlaySettingsCard")
        overlay_layout = QHBoxLayout(overlay_card)
        overlay_layout.setContentsMargins(16, 10, 16, 10)
        overlay_layout.setSpacing(12)
        self.overlay_visible_checkbox = QCheckBox("Overlay 표시")
        self.overlay_visible_checkbox.setChecked(True)
        self.overlay_font_size = QSpinBox()
        self.overlay_font_size.setRange(16, 52)
        self.overlay_font_size.setValue(28)
        self.overlay_opacity = QSlider(Qt.Orientation.Horizontal)
        self.overlay_opacity.setRange(20, 95)
        self.overlay_opacity.setValue(72)
        self.overlay_bold_checkbox = QCheckBox("굵게")
        self.overlay_bold_checkbox.setChecked(True)
        self.overlay_text_color_combo = QComboBox()
        self.overlay_text_color_combo.setObjectName("overlayTextColor")
        self.overlay_text_color_combo.addItem("흰색", "#FFFFFF")
        self.overlay_text_color_combo.addItem("노란색", "#FDE047")
        self.overlay_text_color_combo.addItem("연두색", "#A7F3D0")
        self.overlay_custom_color_button = QPushButton("사용자 지정 색상")
        self.overlay_custom_color_button.setObjectName("overlayCustomColor")
        overlay_layout.addWidget(self.overlay_visible_checkbox)
        overlay_layout.addWidget(QLabel("글자 크기"))
        overlay_layout.addWidget(self.overlay_font_size)
        overlay_layout.addWidget(QLabel("배경 투명도"))
        overlay_layout.addWidget(self.overlay_opacity, 1)
        overlay_layout.addWidget(self.overlay_bold_checkbox)
        overlay_layout.addWidget(QLabel("글자 색상"))
        overlay_layout.addWidget(self.overlay_text_color_combo)
        overlay_layout.addWidget(self.overlay_custom_color_button)
        root.addWidget(overlay_card)

        subtitle_card = QFrame()
        subtitle_card.setObjectName("subtitleCard")
        subtitle_layout = QVBoxLayout(subtitle_card)
        subtitle_layout.setContentsMargins(1, 1, 1, 1)
        self.subtitle_model = SubtitleFeedModel(self)
        self.subtitle_feed = SubtitleFeedView()
        self.subtitle_feed.setModel(self.subtitle_model)
        self.subtitle_model.entry_upserted.connect(
            self.subtitle_feed.keep_latest_visible
        )
        subtitle_layout.addWidget(self.subtitle_feed)
        root.addWidget(subtitle_card, 1)

        status_card = QFrame()
        status_card.setObjectName("statusCard")
        status_layout = QHBoxLayout(status_card)
        status_layout.setContentsMargins(16, 11, 16, 11)
        self.status_label = QLabel("대기 중")
        self.status_label.setObjectName("statusText")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumWidth(0)
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.latency_label = QLabel("최근 latency: —")
        self.latency_label.setObjectName("latencyText")
        status_layout.addWidget(self.status_label, 1)
        status_layout.addWidget(self.latency_label)
        root.addWidget(status_card)

        self.start_button.clicked.connect(self.start_captioning)
        self.stop_button.clicked.connect(self.stop_captioning)
        self.refresh_device_button.clicked.connect(self.refresh_output_devices)
        self.controller.server_state.connect(self.set_server_state)
        self.controller.device_state.connect(self.device_label.setText)
        self.controller.status.connect(self.set_status_message)
        self.controller.subtitle_event.connect(self.apply_subtitle_event)
        self.controller.latency_updated.connect(self.set_latency)
        self.controller.error.connect(self.show_error)
        self.controller.state_changed.connect(self.apply_controller_state)
        self.controller.run_finished.connect(self._run_finished)
        self.overlay_visible_checkbox.toggled.connect(self.set_overlay_visible)
        self.overlay_font_size.valueChanged.connect(self.apply_overlay_style)
        self.overlay_opacity.valueChanged.connect(self.apply_overlay_style)
        self.overlay_bold_checkbox.toggled.connect(self.apply_overlay_style)
        self.overlay_text_color_combo.currentIndexChanged.connect(self.apply_overlay_style)
        self.overlay_custom_color_button.clicked.connect(self.select_custom_overlay_color)

        self.apply_controller_state(self.controller.state)
        self.setStyleSheet(APP_STYLE)

    @Slot()
    def start_captioning(self) -> None:
        server_url = self.server_url_edit.text().strip()
        if not server_url:
            self.show_error("RunPod 서버 URL을 입력해주세요.")
            return
        if self.controller.state != "idle":
            return
        if not self._has_available_output_devices:
            self.show_error(
                self._output_device_error
                or "사용 가능한 Windows 출력 장치가 없습니다. 장치를 연결한 뒤 새로고침하세요."
            )
            return
        self.subtitle_model.clear()
        self.latency_label.setText("최근 latency: —")
        if self.controller.start(
            replace(
                self._launch_config,
                server_url=server_url,
                speaker_mode=SpeakerMode(self.speaker_mode_combo.currentData()),
                output_device_id=self.output_device_combo.currentData(),
            )
        ):
            self._ensure_overlay()
            if self.overlay_visible_checkbox.isChecked() and self.overlay is not None:
                self.overlay.show()
                self.overlay.raise_()

    @Slot()
    def stop_captioning(self) -> None:
        self.controller.stop()
        self._hide_and_clear_overlay()
        self.subtitle_model.clear()

    @Slot(bool, str)
    def set_server_state(self, connected: bool, text: str) -> None:
        color = "#16a34a" if connected else "#dc2626"
        self.server_state_label.setText(f"● {text}")
        self._set_indicator(self.server_state_label, color)

    @Slot(dict)
    def apply_subtitle_event(self, event: dict[str, Any]) -> None:
        self.subtitle_model.upsert_event(event)

    @Slot(bool)
    def set_overlay_visible(self, visible: bool) -> None:
        if visible and self.controller.state in {"starting", "running"}:
            overlay = self._ensure_overlay()
            overlay.show()
            overlay.raise_()
        elif self.overlay is not None:
            self.overlay.hide()

    def apply_overlay_style(self, *_args: object) -> None:
        if self.overlay is not None:
            self.overlay.apply_style(self._overlay_style())

    @Slot()
    def select_custom_overlay_color(self) -> None:
        """Apply a user-selected color through the existing overlay-style path."""
        color = QColorDialog.getColor(
            QColor(str(self.overlay_text_color_combo.currentData())),
            self,
            "자막 글자 색상 선택",
        )
        if not color.isValid():
            return

        color_name = color.name().upper()
        custom_index = self.overlay_text_color_combo.findText("사용자 지정 색상")
        if custom_index < 0:
            self.overlay_text_color_combo.addItem("사용자 지정 색상", color_name)
            custom_index = self.overlay_text_color_combo.count() - 1
        else:
            self.overlay_text_color_combo.setItemData(custom_index, color_name)
        if self.overlay_text_color_combo.currentIndex() != custom_index:
            self.overlay_text_color_combo.setCurrentIndex(custom_index)
        else:
            # Updating an existing custom entry does not emit a selection signal.
            # Refresh visible PARTIAL and FINAL rows immediately in that case.
            self.apply_overlay_style()

    def _overlay_style(self) -> OverlayStyle:
        return OverlayStyle(
            font_size=self.overlay_font_size.value(),
            background_opacity=self.overlay_opacity.value(),
            bold=self.overlay_bold_checkbox.isChecked(),
            text_color=str(self.overlay_text_color_combo.currentData()),
        )

    def _ensure_overlay(self) -> SubtitleOverlay:
        if self.overlay is None:
            self.overlay = SubtitleOverlay()
            self.subtitle_model.entries_changed.connect(self.overlay.set_entries)
        self.overlay.set_keep_empty_visible(self.controller.state in {"starting", "running"})
        self.overlay.apply_style(self._overlay_style())
        self.overlay.set_entries(self.subtitle_model.entries)
        return self.overlay

    def _hide_and_clear_overlay(self) -> None:
        if self.overlay is not None:
            self.overlay.hide()
            self.overlay.set_keep_empty_visible(False)
            self.overlay.clear_entries()

    @Slot(float)
    def set_latency(self, milliseconds: float) -> None:
        self.latency_label.setText(f"최근 latency: {milliseconds:.0f} ms")

    @Slot(str)
    def show_error(self, message: str) -> None:
        text = f"오류: {message}"
        self.status_label.setText(text)
        self.status_label.setToolTip(text)
        self.status_label.setStyleSheet("color: #b91c1c;")

    @Slot(str)
    def set_status_message(self, message: str) -> None:
        self.status_label.setToolTip("")
        self.status_label.setText(message)

    @Slot(str)
    def apply_controller_state(self, state: str) -> None:
        active = state != "idle"
        self.start_button.setEnabled(not active and self._has_available_output_devices)
        self.stop_button.setEnabled(state in {"starting", "running"})
        self.server_url_edit.setEnabled(not active)
        self.speaker_mode_combo.setEnabled(not active)
        self.output_device_combo.setEnabled(not active and self._has_available_output_devices)
        self.refresh_device_button.setEnabled(not active)
        if state == "starting":
            self.status_label.setStyleSheet("color: #1d4ed8;")
            self.set_status_message("서버와 오디오 장치 확인 중…")
        elif state == "stopping":
            self.stop_button.setEnabled(False)
            self.subtitle_model.clear()
            self._hide_and_clear_overlay()
        elif state == "idle" and not self.status_label.text().startswith("오류:"):
            self.status_label.setStyleSheet("color: #334155;")

    @Slot()
    def refresh_output_devices(self) -> None:
        """Refresh idle-only device choices without changing capture in flight."""
        if self.controller.state != "idle":
            return
        self._has_available_output_devices = self._refresh_output_devices(
            announce_selection_change=True,
        )
        self.apply_controller_state(self.controller.state)

    def _refresh_output_devices(self, *, announce_selection_change: bool) -> bool:
        """Populate stable endpoint IDs and retain a surviving explicit choice."""
        previous_device_id = self.output_device_combo.currentData()
        self.output_device_combo.clear()
        self.output_device_combo.addItem("Windows 기본 출력 장치", None)
        try:
            import soundcard as sc

            devices = list_output_devices(sc)
        except Exception as exc:
            self._output_device_error = (
                f"오디오 장치 목록을 가져올 수 없습니다: {exc}. 새로고침 후 다시 시도하세요."
            )
            self.device_label.setText(self._output_device_error)
            self.output_device_combo.setEnabled(False)
            self.output_device_combo.setToolTip(self._output_device_error)
            return False
        if not devices:
            self._output_device_error = (
                "사용 가능한 Windows 출력 장치가 없습니다. 장치를 연결한 뒤 새로고침하세요."
            )
            self.device_label.setText(self._output_device_error)
            self.output_device_combo.setEnabled(False)
            self.output_device_combo.setToolTip(self._output_device_error)
            return False
        self._output_device_error = ""
        self.output_device_combo.setToolTip("")
        for device in devices:
            suffix = "기본" if device.is_default else device.device_id[-12:]
            self.output_device_combo.addItem(f"{device.name} ({suffix})", device.device_id)
            self.output_device_combo.setItemData(
                self.output_device_combo.count() - 1,
                f"WASAPI ID: {device.device_id}",
                Qt.ItemDataRole.ToolTipRole,
            )
        if previous_device_id is not None:
            retained_index = self.output_device_combo.findData(previous_device_id)
            if retained_index >= 0:
                self.output_device_combo.setCurrentIndex(retained_index)
                if announce_selection_change:
                    self.device_label.setText("선택한 오디오 장치를 유지하고 목록을 새로고침했습니다.")
            else:
                self.output_device_combo.setCurrentIndex(0)
                if announce_selection_change:
                    self.device_label.setText(
                        "선택한 오디오 장치가 사라져 Windows 기본 출력 장치로 변경했습니다."
                    )
        elif announce_selection_change:
            self.device_label.setText("오디오 장치 목록을 새로고침했습니다.")
        return True

    @Slot(bool, str)
    def _run_finished(self, success: bool, message: str) -> None:
        self.subtitle_model.clear()
        self._hide_and_clear_overlay()
        if success:
            self.status_label.setStyleSheet("color: #334155;")
            self.status_label.setText("대기 중")
        elif not self.status_label.text().startswith("오류:"):
            self.show_error(message)
        if self._close_when_finished:
            self._close_when_finished = False
            self.close()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt API)
        if self.controller.state == "idle":
            self._hide_and_clear_overlay()
            if self.overlay is not None:
                self.overlay.close()
                self.overlay = None
            event.accept()
            return
        self._close_when_finished = True
        self.controller.stop()
        event.ignore()

    @staticmethod
    def _set_indicator(label: QLabel, color: str) -> None:
        label.setStyleSheet(f"color: {color}; font-weight: 700;")


def parse_launch_options(argv: list[str]) -> tuple[LiveCaptionConfig, list[str]]:
    parser = argparse.ArgumentParser(description="VOXLUX GUI live A/B test options", allow_abbrev=False)
    parser.add_argument("--window-seconds", type=int, default=3)
    parser.add_argument("--stride-seconds", type=int, default=2)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--latency-diagnostics", action="store_true")
    # Preserve Qt's existing command-line options (for example -platform).
    args, qt_args = parser.parse_known_args(argv)
    try:
        config = LiveCaptionConfig(
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            evidence_dir=args.evidence_dir,
            latency_diagnostics=args.latency_diagnostics,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return config, qt_args


def main(argv: list[str] | None = None) -> int:
    config, qt_args = parse_launch_options(sys.argv[1:] if argv is None else argv)
    try:
        if config.evidence_dir is not None:
            configure_gui_runtime(diagnostic_dir=config.evidence_dir)
        else:
            configure_gui_runtime()
        app = QApplication.instance() or QApplication([sys.argv[0], *qt_args])
        window = MainWindow(config=config)
        window.show()
        return app.exec()
    except BaseException:
        # pythonw has no console for an uncaught GUI startup exception.
        LOGGER.exception("VOXLUX GUI startup failed")
        raise
    finally:
        shutdown_gui_runtime()


if __name__ == "__main__":
    raise SystemExit(main())
