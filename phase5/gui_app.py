from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

# See live_caption_controller.py: this import must precede PySide6 on Python 3.10.
from typing_extensions import Self as _TypingExtensionsSelf  # noqa: F401

from PySide6.QtCore import QAbstractListModel, QModelIndex, QSize, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

try:
    from .live_caption_controller import (
        DEFAULT_REMOTE_SERVER_URL,
        LiveCaptionConfig,
        LiveCaptionController,
    )
except ImportError:  # Direct execution: python phase5/gui_app.py
    from live_caption_controller import (  # type: ignore[no-redef]
        DEFAULT_REMOTE_SERVER_URL,
        LiveCaptionConfig,
        LiveCaptionController,
    )


APP_STYLE = """
QMainWindow, QWidget#centralWidget { background: #f4f7fb; color: #172033; }
QLabel#title { color: #111827; font-size: 30px; font-weight: 700; }
QLabel#hint { color: #64748b; font-size: 12px; }
QFrame#infoCard, QFrame#subtitleCard, QFrame#statusCard {
    background: white;
    border: 1px solid #dbe3ef;
    border-radius: 12px;
}
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


@dataclass
class SubtitleFeedEntry:
    speaker_id: str
    utterance_id: int
    sequence: int
    status: str
    text: str
    timestamp: int | float | None = None


class SubtitleFeedModel(QAbstractListModel):
    """Bounded chronological utterance model; speaker identity stays internal."""

    entry_upserted = Signal(int, bool)
    MAX_ENTRIES = 80
    SpeakerRole = Qt.ItemDataRole.UserRole + 1
    UtteranceRole = Qt.ItemDataRole.UserRole + 2
    SequenceRole = Qt.ItemDataRole.UserRole + 3
    StatusRole = Qt.ItemDataRole.UserRole + 4

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: list[SubtitleFeedEntry] = []
        self._rows_by_utterance: dict[tuple[str, int], int] = {}

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
        self._entries.clear()
        self._rows_by_utterance.clear()
        self.endResetModel()

    def upsert_event(self, event: dict[str, Any]) -> bool:
        speaker_id = self._speaker_id(event.get("speaker"))
        utterance_id = event.get("utterance_id")
        sequence = event.get("sequence")
        status = event.get("status")
        text = event.get("text")
        if (
            speaker_id is None
            or not isinstance(utterance_id, int)
            or not isinstance(sequence, int)
            or status not in {"partial", "final"}
            or not isinstance(text, str)
            or not text.strip()
        ):
            return False

        key = (speaker_id, utterance_id)
        existing_row = self._rows_by_utterance.get(key)
        if existing_row is not None:
            existing = self._entries[existing_row]
            if sequence < existing.sequence or existing.status == "final":
                return False
            self._entries[existing_row] = SubtitleFeedEntry(
                speaker_id=speaker_id,
                utterance_id=utterance_id,
                sequence=sequence,
                status=status,
                text=text.strip(),
                timestamp=event.get("timestamp"),
            )
            model_index = self.index(existing_row)
            self.dataChanged.emit(
                model_index,
                model_index,
                [Qt.ItemDataRole.DisplayRole, self.SequenceRole, self.StatusRole],
            )
            self.entry_upserted.emit(existing_row, False)
            return True

        row = len(self._entries)
        self.beginInsertRows(QModelIndex(), row, row)
        self._entries.append(
            SubtitleFeedEntry(
                speaker_id=speaker_id,
                utterance_id=utterance_id,
                sequence=sequence,
                status=status,
                text=text.strip(),
                timestamp=event.get("timestamp"),
            )
        )
        self._rows_by_utterance[key] = row
        self.endInsertRows()
        self._trim_old_entries()
        inserted_row = self._rows_by_utterance[key]
        self.entry_upserted.emit(inserted_row, True)
        return True

    def entry_at(self, row: int) -> SubtitleFeedEntry:
        return self._entries[row]

    @staticmethod
    def _speaker_id(value: Any) -> str | None:
        if value in (0, "speaker_0"):
            return "speaker_0"
        if value in (1, "speaker_1"):
            return "speaker_1"
        return None

    def _trim_old_entries(self) -> None:
        overflow = len(self._entries) - self.MAX_ENTRIES
        if overflow <= 0:
            return
        self.beginRemoveRows(QModelIndex(), 0, overflow - 1)
        del self._entries[:overflow]
        self.endRemoveRows()
        self._rows_by_utterance = {
            (entry.speaker_id, entry.utterance_id): row
            for row, entry in enumerate(self._entries)
        }


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
    ) -> None:
        super().__init__(parent)
        self.controller = controller or LiveCaptionController()
        self._close_when_finished = False
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
        self.device_label = QLabel("자막 시작 시 WASAPI loopback 장치를 확인합니다.")
        self.device_label.setObjectName("deviceState")
        self.device_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        device_row.addWidget(device_title)
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
        self.latency_label = QLabel("최근 latency: —")
        self.latency_label.setObjectName("latencyText")
        status_layout.addWidget(self.status_label, 1)
        status_layout.addWidget(self.latency_label)
        root.addWidget(status_card)

        self.start_button.clicked.connect(self.start_captioning)
        self.stop_button.clicked.connect(self.stop_captioning)
        self.controller.server_state.connect(self.set_server_state)
        self.controller.device_state.connect(self.device_label.setText)
        self.controller.status.connect(self.status_label.setText)
        self.controller.subtitle_event.connect(self.apply_subtitle_event)
        self.controller.latency_updated.connect(self.set_latency)
        self.controller.error.connect(self.show_error)
        self.controller.state_changed.connect(self.apply_controller_state)
        self.controller.run_finished.connect(self._run_finished)

        self.setStyleSheet(APP_STYLE)

    @Slot()
    def start_captioning(self) -> None:
        server_url = self.server_url_edit.text().strip()
        if not server_url:
            self.show_error("RunPod 서버 URL을 입력해주세요.")
            return
        self.subtitle_model.clear()
        self.latency_label.setText("최근 latency: —")
        self.controller.start(LiveCaptionConfig(server_url=server_url))

    @Slot()
    def stop_captioning(self) -> None:
        self.controller.stop()

    @Slot(bool, str)
    def set_server_state(self, connected: bool, text: str) -> None:
        color = "#16a34a" if connected else "#dc2626"
        self.server_state_label.setText(f"● {text}")
        self._set_indicator(self.server_state_label, color)

    @Slot(dict)
    def apply_subtitle_event(self, event: dict[str, Any]) -> None:
        self.subtitle_model.upsert_event(event)

    @Slot(float)
    def set_latency(self, milliseconds: float) -> None:
        self.latency_label.setText(f"최근 latency: {milliseconds:.0f} ms")

    @Slot(str)
    def show_error(self, message: str) -> None:
        self.status_label.setText(f"오류: {message}")
        self.status_label.setStyleSheet("color: #b91c1c;")

    @Slot(str)
    def apply_controller_state(self, state: str) -> None:
        active = state != "idle"
        self.start_button.setEnabled(not active)
        self.stop_button.setEnabled(state in {"starting", "running"})
        self.server_url_edit.setEnabled(not active)
        if state == "starting":
            self.status_label.setStyleSheet("color: #1d4ed8;")
            self.status_label.setText("서버와 오디오 장치 확인 중…")
        elif state == "stopping":
            self.stop_button.setEnabled(False)
        elif state == "idle" and not self.status_label.text().startswith("오류:"):
            self.status_label.setStyleSheet("color: #334155;")

    @Slot(bool, str)
    def _run_finished(self, success: bool, message: str) -> None:
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
            event.accept()
            return
        self._close_when_finished = True
        self.controller.stop()
        event.ignore()

    @staticmethod
    def _set_indicator(label: QLabel, color: str) -> None:
        label.setStyleSheet(f"color: {color}; font-weight: 700;")


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
