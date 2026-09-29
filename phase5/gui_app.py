from __future__ import annotations

import sys
from typing import Any

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QTextEdit,
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
QFrame#infoCard, QFrame#speakerCard, QFrame#statusCard {
    background: white;
    border: 1px solid #dbe3ef;
    border-radius: 12px;
}
QLabel#fieldTitle { color: #526078; font-weight: 600; }
QLabel#speakerTitle { font-size: 20px; font-weight: 700; color: #1d4ed8; }
QLabel#speakerStatus { color: #64748b; font-size: 12px; }
QLineEdit {
    background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 7px;
    padding: 8px 10px; color: #1e293b;
}
QTextEdit {
    background: #fbfdff; border: 0; border-top: 1px solid #e2e8f0;
    padding: 12px; color: #0f172a;
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


class SpeakerPanel(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("speakerCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 18)
        layout.setSpacing(9)

        heading = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("speakerTitle")
        self.event_status = QLabel("대기 중")
        self.event_status.setObjectName("speakerStatus")
        heading.addWidget(title_label)
        heading.addStretch(1)
        heading.addWidget(self.event_status)
        layout.addLayout(heading)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlaceholderText("자막을 기다리는 중입니다.")
        self.text.setMinimumHeight(170)
        self.text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        font = QFont()
        font.setPointSize(22)
        font.setWeight(QFont.Weight.DemiBold)
        self.text.setFont(font)
        layout.addWidget(self.text, 1)

    def update_event(self, text: str, status: str) -> None:
        self.text.setPlainText(text)
        cursor = self.text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.text.setTextCursor(cursor)
        self.event_status.setText("확정" if status == "final" else "인식 중")

    def reset(self) -> None:
        self.text.clear()
        self.event_status.setText("대기 중")


class MainWindow(QMainWindow):
    def __init__(
        self,
        controller: LiveCaptionController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller or LiveCaptionController()
        self._last_sequence = {0: 0, 1: 0}
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

        self.speaker_a = SpeakerPanel("Speaker A")
        self.speaker_a.text.setObjectName("speakerAText")
        self.speaker_b = SpeakerPanel("Speaker B")
        self.speaker_b.text.setObjectName("speakerBText")
        root.addWidget(self.speaker_a, 1)
        root.addWidget(self.speaker_b, 1)

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
        self.speaker_a.reset()
        self.speaker_b.reset()
        self._last_sequence = {0: 0, 1: 0}
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
        raw_speaker = event.get("speaker")
        if raw_speaker in (0, "speaker_0"):
            speaker = 0
            panel = self.speaker_a
        elif raw_speaker in (1, "speaker_1"):
            speaker = 1
            panel = self.speaker_b
        else:
            return

        sequence = event.get("sequence", 0)
        if isinstance(sequence, int) and sequence < self._last_sequence[speaker]:
            return
        if isinstance(sequence, int):
            self._last_sequence[speaker] = sequence
        display_text = event.get("assembled_text") or event.get("text") or ""
        panel.update_event(str(display_text), str(event.get("status", "partial")))

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
