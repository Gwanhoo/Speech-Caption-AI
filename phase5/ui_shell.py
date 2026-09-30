"""Standalone PySide6 UI shell; deliberately has no server/audio integration."""
from __future__ import annotations

import sys
from dataclasses import dataclass

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QDialog, QFormLayout, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QSlider, QVBoxLayout, QWidget,
)


@dataclass
class CaptionStyle:
    font_size: int = 26
    bold: bool = True
    text_color: str = "#FFFFFF"
    opacity: int = 75


class SubtitleOverlay(QWidget):
    """Independent, draggable, always-on-top subtitle overlay."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.style_config = CaptionStyle()
        self._drag_position: QPoint | None = None
        self.setWindowTitle("AI 실시간 자막")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(900, 160)
        self.caption = QLabel("자막 Bar가 생성되었습니다.", self)
        self.caption.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.caption.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.addWidget(self.caption)
        self.apply_style()

    def apply_style(self) -> None:
        font = QFont()
        font.setPointSize(self.style_config.font_size)
        font.setBold(self.style_config.bold)
        self.caption.setFont(font)
        alpha = self.style_config.opacity / 100
        self.caption.setStyleSheet(
            f"QLabel {{ color: {self.style_config.text_color}; "
            f"background-color: rgba(10, 15, 25, {alpha}); "
            "border: 1px solid rgba(255, 255, 255, 70); border-radius: 18px; "
            "padding: 18px 25px; }"
        )

    def update_subtitle(self, text: str) -> None:
        text = text.strip()
        if text:
            self.caption.setText(text)

    def update_style(self, config: CaptionStyle) -> None:
        self.style_config = config
        self.apply_style()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton and self._drag_position is not None:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_position = None
        super().mouseReleaseEvent(event)


class SettingsDialog(QDialog):
    settings_changed = Signal(object)

    def __init__(self, config: CaptionStyle, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("자막 설정")
        self.setFixedWidth(400)
        self.config = CaptionStyle(config.font_size, config.bold, config.text_color, config.opacity)
        layout = QFormLayout(self)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(18)

        self.font_slider = QSlider(Qt.Orientation.Horizontal)
        self.font_slider.setRange(16, 48)
        self.font_slider.setValue(self.config.font_size)
        self.font_value = QLabel(str(self.config.font_size))
        font_row = QWidget()
        font_layout = QHBoxLayout(font_row)
        font_layout.setContentsMargins(0, 0, 0, 0)
        font_layout.addWidget(self.font_slider)
        font_layout.addWidget(self.font_value)
        self.font_slider.valueChanged.connect(lambda value: self.font_value.setText(str(value)))
        layout.addRow("글자 크기", font_row)

        self.bold_checkbox = QCheckBox("굵게 표시")
        self.bold_checkbox.setChecked(self.config.bold)
        layout.addRow("글자 굵기", self.bold_checkbox)
        self.color_button = QPushButton("색상 선택")
        self.color_button.clicked.connect(self.select_color)
        layout.addRow("글자 색상", self.color_button)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(self.config.opacity)
        self.opacity_value = QLabel(f"{self.config.opacity}%")
        opacity_row = QWidget()
        opacity_layout = QHBoxLayout(opacity_row)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addWidget(self.opacity_value)
        self.opacity_slider.valueChanged.connect(lambda value: self.opacity_value.setText(f"{value}%"))
        layout.addRow("배경 불투명도", opacity_row)

        self.save_button = QPushButton("적용")
        self.save_button.setMinimumHeight(40)
        self.save_button.clicked.connect(self.save_settings)
        layout.addRow(self.save_button)

    def select_color(self) -> None:
        color = QColorDialog.getColor(QColor(self.config.text_color), self, "자막 글자색 선택")
        if color.isValid():
            self.config.text_color = color.name()

    def save_settings(self) -> None:
        self.config.font_size = self.font_slider.value()
        self.config.bold = self.bold_checkbox.isChecked()
        self.config.opacity = self.opacity_slider.value()
        self.settings_changed.emit(self.config)
        self.accept()


class MainWindow(QMainWindow):
    start_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.overlay: SubtitleOverlay | None = None
        self.setWindowTitle("AI 실시간 자막")
        self.setFixedSize(440, 360)
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(30, 28, 30, 25)
        root.setSpacing(16)
        title = QLabel("AI 실시간 자막")
        title.setObjectName("title")
        description = QLabel("PC에서 재생되는 음성을 실시간 자막으로 표시합니다.")
        description.setObjectName("description")
        description.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(description)
        row1, row2 = QHBoxLayout(), QHBoxLayout()
        self.start_button = QPushButton("▶  시작")
        self.stop_button = QPushButton("■  중지")
        self.settings_button = QPushButton("⚙  설정")
        self.create_button = QPushButton("＋  생성")
        for button in (self.start_button, self.stop_button, self.settings_button, self.create_button):
            button.setMinimumHeight(58)
        row1.addWidget(self.start_button)
        row1.addWidget(self.stop_button)
        row2.addWidget(self.settings_button)
        row2.addWidget(self.create_button)
        root.addLayout(row1)
        root.addLayout(row2)
        root.addStretch()
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("상태"))
        status_layout.addStretch()
        self.status_label = QLabel("● 자막 Bar를 생성해주세요")
        self.status_label.setObjectName("status")
        status_layout.addWidget(self.status_label)
        root.addLayout(status_layout)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.create_button.clicked.connect(self.create_overlay)
        self.start_button.clicked.connect(self.start_caption)
        self.stop_button.clicked.connect(self.stop_caption)
        self.settings_button.clicked.connect(self.open_settings)
        self.setStyleSheet("""
            QMainWindow { background: #F7F9FC; }
            QLabel { color: #182230; font-size: 14px; }
            QLabel#title { font-size: 26px; font-weight: 700; }
            QLabel#description { color: #667085; font-size: 13px; }
            QLabel#status { color: #667085; font-size: 13px; font-weight: 600; }
            QPushButton { background: #FFFFFF; color: #182230; border: 1px solid #D0D5DD; border-radius: 11px; font-size: 15px; font-weight: 600; padding: 10px; }
            QPushButton:hover { border: 1px solid #1570EF; background: #F5F9FF; }
            QPushButton:pressed { background: #EAF2FF; }
            QPushButton:disabled { color: #98A2B3; background: #EAECF0; border: 1px solid #EAECF0; }
        """)

    def create_overlay(self) -> None:
        if self.overlay is None:
            self.overlay = SubtitleOverlay()
        self.overlay.show()
        self.overlay.raise_()
        self.start_button.setEnabled(True)
        self.status_label.setText("● 자막 Bar 준비됨")

    def start_caption(self) -> None:
        if self.overlay is None:
            return
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.create_button.setEnabled(False)
        self.status_label.setText("● 실행 중")
        self.overlay.update_subtitle("오늘 저녁에 뭐 먹을지 아직 못 정했는데 혹시 먹고 싶은 거 있어?")
        self.start_requested.emit()

    def stop_caption(self) -> None:
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(self.overlay is not None)
        self.create_button.setEnabled(True)
        self.status_label.setText("● 중지됨")
        self.stop_requested.emit()

    def open_settings(self) -> None:
        config = self.overlay.style_config if self.overlay is not None else CaptionStyle()
        dialog = SettingsDialog(config, self)
        dialog.settings_changed.connect(self.apply_overlay_settings)
        dialog.exec()

    def apply_overlay_settings(self, config: CaptionStyle) -> None:
        if self.overlay is not None:
            self.overlay.update_style(config)

    def update_subtitle(self, text: str) -> None:
        if self.overlay is not None:
            self.overlay.update_subtitle(text)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AI 실시간 자막")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
