"""Standalone PySide6 GUI for human review of the VOXLUX A4 clips."""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import html
from pathlib import Path
import sys

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QCloseEvent
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton, QSlider, QSplitter, QTextBrowser,
    QVBoxLayout, QWidget,
)

from review_store import ReviewStore


DATA_DIR = Path(__file__).resolve().parent
ROLE_SAMPLE_ID = Qt.ItemDataRole.UserRole


def format_time(milliseconds: int) -> str:
    seconds = max(0, milliseconds) / 1000
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def diff_html(left: str, right: str) -> tuple[str, str]:
    left_parts: list[str] = []
    right_parts: list[str] = []
    for opcode, i1, i2, j1, j2 in SequenceMatcher(None, left, right).get_opcodes():
        left_text = html.escape(left[i1:i2]).replace("\n", "<br>")
        right_text = html.escape(right[j1:j2]).replace("\n", "<br>")
        if opcode == "equal":
            left_parts.append(left_text); right_parts.append(right_text)
        else:
            if left_text:
                left_parts.append(f'<span style="background:#ffe0a6;color:#7a3d00;">{left_text}</span>')
            if right_text:
                right_parts.append(f'<span style="background:#d7e8ff;color:#003c78;">{right_text}</span>')
    style = "font-size:15px; line-height:1.45; white-space:pre-wrap;"
    return f'<div style="{style}">{"".join(left_parts)}</div>', f'<div style="{style}">{"".join(right_parts)}</div>'


class ReviewWindow(QMainWindow):
    def __init__(self, store: ReviewStore, *, audio_enabled: bool = True) -> None:
        super().__init__()
        self.store = store
        self.audio_enabled = audio_enabled
        self.current_id: str | None = None
        self.listened_ids: set[str] = set()
        self.dirty = False
        self.loading = False
        self.selection_guard = False
        self.setWindowTitle("VOXLUX A4 오디오 정답 검수")
        self.resize(1240, 820)
        self._build_ui()
        self.player: QMediaPlayer | None = None
        self.audio_output: QAudioOutput | None = None
        if audio_enabled:
            self._build_player()
        else:
            self._set_audio_controls_enabled(False)
        self.refresh_list(preferred_id=self.store.order[0])

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        header = QHBoxLayout()
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("font-size:17px;font-weight:600;")
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("전체", "all")
        self.filter_combo.addItem("pending만", "pending")
        self.filter_combo.addItem("verified만", "verified")
        self.filter_combo.currentIndexChanged.connect(self._filter_changed)
        self.save_button = QPushButton("저장")
        self.save_button.clicked.connect(self.save_current)
        self.evaluate_button = QPushButton("검수 완료 후 CER/WER 평가")
        self.evaluate_button.clicked.connect(self.evaluate)
        header.addWidget(self.progress_label)
        header.addStretch(1)
        header.addWidget(QLabel("필터:")); header.addWidget(self.filter_combo)
        header.addWidget(self.save_button); header.addWidget(self.evaluate_button)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.clip_list = QListWidget()
        self.clip_list.setMinimumWidth(300)
        self.clip_list.currentItemChanged.connect(self._selection_changed)
        splitter.addWidget(self.clip_list)
        detail = QWidget(); detail_layout = QVBoxLayout(detail)
        self.id_label = QLabel("클립 미선택")
        self.id_label.setStyleSheet("font-size:18px;font-weight:600;")
        self.status_label = QLabel()
        detail_layout.addWidget(self.id_label); detail_layout.addWidget(self.status_label)

        playback = QGroupBox("오디오 재생")
        playback_layout = QGridLayout(playback)
        self.play_button = QPushButton("재생")
        self.pause_button = QPushButton("일시정지")
        self.stop_button = QPushButton("정지")
        self.play_button.clicked.connect(self.play); self.pause_button.clicked.connect(self.pause); self.stop_button.clicked.connect(self.stop)
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0); self.position_slider.sliderMoved.connect(self.seek)
        self.time_label = QLabel("00:00.0 / 00:00.0")
        self.loop_check = QCheckBox("구간 반복")
        self.speed_combo = QComboBox()
        for speed in (0.75, 1.0, 1.25): self.speed_combo.addItem(f"{speed:.2f}x", speed)
        self.speed_combo.setCurrentIndex(1); self.speed_combo.currentIndexChanged.connect(self.set_speed)
        playback_layout.addWidget(self.play_button, 0, 0); playback_layout.addWidget(self.pause_button, 0, 1)
        playback_layout.addWidget(self.stop_button, 0, 2); playback_layout.addWidget(self.loop_check, 0, 3)
        playback_layout.addWidget(QLabel("속도"), 0, 4); playback_layout.addWidget(self.speed_combo, 0, 5)
        playback_layout.addWidget(self.position_slider, 1, 0, 1, 5); playback_layout.addWidget(self.time_label, 1, 5)
        detail_layout.addWidget(playback)

        comparison = QGroupBox("STT 원본 비교 — 색상 부분이 서로 다른 구간")
        comparison_layout = QGridLayout(comparison)
        comparison_layout.addWidget(QLabel("large-v3-turbo"), 0, 0)
        comparison_layout.addWidget(QLabel("large-v3"), 0, 1)
        self.turbo_view = QTextBrowser(); self.large_view = QTextBrowser()
        self.turbo_view.setMinimumHeight(140); self.large_view.setMinimumHeight(140)
        comparison_layout.addWidget(self.turbo_view, 1, 0); comparison_layout.addWidget(self.large_view, 1, 1)
        self.copy_turbo = QPushButton("Turbo 결과를 정답란에 복사")
        self.copy_large = QPushButton("Large-v3 결과를 정답란에 복사")
        self.copy_turbo.clicked.connect(lambda: self._copy_model("turbo_transcript"))
        self.copy_large.clicked.connect(lambda: self._copy_model("large_v3_transcript"))
        comparison_layout.addWidget(self.copy_turbo, 2, 0); comparison_layout.addWidget(self.copy_large, 2, 1)
        detail_layout.addWidget(comparison)

        answer = QGroupBox("사람 검수 정답")
        answer_layout = QVBoxLayout(answer)
        self.reference_edit = QPlainTextEdit(); self.reference_edit.setPlaceholderText("오디오를 듣고 실제 발화를 입력하세요.")
        self.reference_edit.setMinimumHeight(105); self.reference_edit.textChanged.connect(self._editor_changed)
        answer_layout.addWidget(self.reference_edit)
        form = QFormLayout()
        self.reviewer_edit = QLineEdit(); self.reviewer_edit.textEdited.connect(self._editor_changed)
        self.notes_edit = QLineEdit(); self.notes_edit.textEdited.connect(self._editor_changed)
        self.reviewed_at_label = QLabel("-")
        form.addRow("검수자", self.reviewer_edit); form.addRow("검수 메모", self.notes_edit); form.addRow("검수 시각", self.reviewed_at_label)
        answer_layout.addLayout(form)
        status_buttons = QHBoxLayout()
        self.pending_button = QPushButton("pending으로 저장")
        self.verify_button = QPushButton("오디오 확인 완료 — verified")
        self.pending_button.clicked.connect(self.mark_pending); self.verify_button.clicked.connect(self.verify)
        status_buttons.addWidget(self.pending_button); status_buttons.addWidget(self.verify_button)
        answer_layout.addLayout(status_buttons)
        detail_layout.addWidget(answer)

        navigation = QHBoxLayout()
        self.previous_button = QPushButton("← 이전")
        self.next_button = QPushButton("다음 →")
        self.previous_button.clicked.connect(lambda: self.move_selection(-1)); self.next_button.clicked.connect(lambda: self.move_selection(1))
        navigation.addWidget(self.previous_button); navigation.addStretch(1); navigation.addWidget(self.next_button)
        detail_layout.addLayout(navigation)
        splitter.addWidget(detail); splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

    def _build_player(self) -> None:
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(1.0)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self._position_changed)
        self.player.durationChanged.connect(self._duration_changed)
        self.player.mediaStatusChanged.connect(self._media_status_changed)
        self.player.errorOccurred.connect(lambda _error, message: QMessageBox.critical(self, "재생 오류", message))

    def _set_audio_controls_enabled(self, enabled: bool) -> None:
        for widget in (self.play_button, self.pause_button, self.stop_button, self.position_slider, self.loop_check, self.speed_combo):
            widget.setEnabled(enabled)

    def visible_ids(self) -> list[str]:
        mode = self.filter_combo.currentData()
        return [sample_id for sample_id in self.store.order if mode == "all" or self.store.get(sample_id)["review_status"] == mode]

    def refresh_list(self, preferred_id: str | None = None) -> None:
        self.selection_guard = True
        self.clip_list.clear()
        for sample_id in self.visible_ids():
            row = self.store.get(sample_id)
            item = QListWidgetItem(f"{row['review_status']:<8}  {sample_id}")
            item.setData(ROLE_SAMPLE_ID, sample_id)
            item.setForeground(QColor("#167347" if row["review_status"] == "verified" else "#8a5a00"))
            self.clip_list.addItem(item)
        self.selection_guard = False
        target = preferred_id if preferred_id in self.visible_ids() else (self.visible_ids()[0] if self.visible_ids() else None)
        if target:
            self._select_id(target)
        else:
            self.current_id = None
            self.id_label.setText("현재 필터에 표시할 클립이 없습니다.")
        self._update_progress()

    def _select_id(self, sample_id: str) -> None:
        for index in range(self.clip_list.count()):
            item = self.clip_list.item(index)
            if item.data(ROLE_SAMPLE_ID) == sample_id:
                self.clip_list.setCurrentRow(index)
                return

    def _selection_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if self.selection_guard or current is None:
            return
        new_id = current.data(ROLE_SAMPLE_ID)
        if self.current_id and new_id != self.current_id and self.dirty:
            if not self.save_current(show_success=False, refresh=False):
                self.selection_guard = True; self._select_id(self.current_id); self.selection_guard = False
                return
        self.load_sample(new_id)

    def load_sample(self, sample_id: str) -> None:
        self.loading = True
        self.current_id = sample_id
        row = self.store.get(sample_id)
        self.id_label.setText(f"{sample_id}  ({row['start_seconds']}–{row['end_seconds']}초)")
        left, right = diff_html(row["turbo_transcript"], row["large_v3_transcript"])
        self.turbo_view.setHtml(left); self.large_view.setHtml(right)
        self.reference_edit.setPlainText(row["reference_text"])
        self.reviewer_edit.setText(row["reviewer"])
        self.notes_edit.setText(row["review_notes"])
        self.reviewed_at_label.setText(row["reviewed_at"] or "-")
        self._update_status_label()
        self.dirty = False
        if self.player:
            self.player.stop(); self.player.setSource(QUrl.fromLocalFile(str(self.store.audio_path(sample_id))))
            self.position_slider.setValue(0); self.time_label.setText("00:00.0 / 00:00.0")
        self.loading = False

    def _editor_changed(self) -> None:
        if self.loading or not self.current_id:
            return
        self.dirty = True
        if self.store.get(self.current_id)["review_status"] == "verified":
            self.store.mark_pending(self.current_id)
            self._update_status_label()

    def _copy_model(self, key: str) -> None:
        if self.current_id:
            self.reference_edit.setPlainText(self.store.get(self.current_id)[key])

    def _persist_widgets(self) -> None:
        if self.current_id:
            self.store.update_draft(self.current_id, self.reference_edit.toPlainText(), self.reviewer_edit.text(), self.notes_edit.text())

    def save_current(self, _checked: bool = False, *, show_success: bool = True, refresh: bool = True) -> bool:
        self._persist_widgets()
        try:
            self.store.save()
        except Exception as exc:
            QMessageBox.critical(self, "저장 실패", str(exc)); return False
        self.dirty = False
        if refresh:
            self.refresh_list(preferred_id=self.current_id)
        if show_success:
            self.statusBar().showMessage(f"저장됨: {self.store.completed_path}", 5000)
        return True

    def mark_pending(self) -> None:
        if not self.current_id: return
        self._persist_widgets(); self.store.mark_pending(self.current_id)
        if self.save_current(show_success=False): self.statusBar().showMessage("pending으로 저장했습니다.", 4000)

    def verify(self) -> None:
        if not self.current_id: return
        self._persist_widgets()
        snapshot = dict(self.store.get(self.current_id))
        try:
            self.store.verify(self.current_id, listened=self.current_id in self.listened_ids)
            self.store.save()
        except Exception as exc:
            self.store.rows[self.current_id] = snapshot
            QMessageBox.warning(self, "verified 처리 거부", str(exc)); return
        self.dirty = False
        self.reviewed_at_label.setText(self.store.get(self.current_id)["reviewed_at"])
        next_id = self._next_pending_id(self.current_id)
        self.refresh_list(preferred_id=next_id or self.current_id)
        self.statusBar().showMessage("verified로 저장했습니다.", 4000)

    def _next_pending_id(self, after_id: str) -> str | None:
        start = self.store.order.index(after_id)
        ordered = self.store.order[start + 1:] + self.store.order[:start]
        return next((sample_id for sample_id in ordered if self.store.get(sample_id)["review_status"] == "pending"), None)

    def _update_status_label(self) -> None:
        if not self.current_id: return
        status = self.store.get(self.current_id)["review_status"]
        heard = "재생함" if self.current_id in self.listened_ids else "아직 재생하지 않음"
        self.status_label.setText(f"상태: {status}  |  {heard}")

    def _update_progress(self) -> None:
        verified, total = self.store.progress()
        self.progress_label.setText(f"검수 진행률: {verified}/{total}")
        self.evaluate_button.setEnabled(verified == total)

    def _filter_changed(self) -> None:
        if self.loading: return
        if self.dirty and not self.save_current(show_success=False, refresh=False): return
        self.refresh_list(preferred_id=self.current_id)

    def move_selection(self, delta: int) -> None:
        count = self.clip_list.count()
        if not count: return
        current = self.clip_list.currentRow()
        self.clip_list.setCurrentRow(max(0, min(count - 1, current + delta)))

    def play(self) -> None:
        if not self.player or not self.current_id: return
        self.listened_ids.add(self.current_id); self._update_status_label(); self.player.play()

    def pause(self) -> None:
        if self.player: self.player.pause()

    def stop(self) -> None:
        if self.player: self.player.stop()

    def seek(self, value: int) -> None:
        if self.player: self.player.setPosition(value)

    def set_speed(self) -> None:
        if self.player: self.player.setPlaybackRate(float(self.speed_combo.currentData()))

    def _position_changed(self, position: int) -> None:
        self.position_slider.setValue(position)
        duration = self.player.duration() if self.player else 0
        self.time_label.setText(f"{format_time(position)} / {format_time(duration)}")

    def _duration_changed(self, duration: int) -> None:
        self.position_slider.setRange(0, max(0, duration))
        position = self.player.position() if self.player else 0
        self.time_label.setText(f"{format_time(position)} / {format_time(duration)}")

    def _media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.MediaStatus.EndOfMedia and self.loop_check.isChecked() and self.player:
            self.player.setPosition(0); self.player.play()

    def evaluate(self) -> None:
        if self.dirty and not self.save_current(show_success=False): return
        try:
            result = self.store.evaluate()
        except Exception as exc:
            QMessageBox.critical(self, "평가 실패", str(exc)); return
        turbo, large = result["models"]["large-v3-turbo"], result["models"]["large-v3"]
        QMessageBox.information(
            self, "평가 완료",
            f"검수 샘플: {result['verified_samples']}\n\n"
            f"large-v3-turbo  CER {turbo['micro_cer']:.2%} / WER {turbo['micro_wer']:.2%}\n"
            f"large-v3        CER {large['micro_cer']:.2%} / WER {large['micro_wer']:.2%}\n\n"
            f"결과: {self.store.evaluation_path}",
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.dirty and not self.save_current(show_success=False, refresh=False):
            answer = QMessageBox.question(self, "저장 실패", "저장하지 않고 종료하시겠습니까?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore(); return
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--smoke-test", action="store_true", help="Load all data/widgets without audio and exit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    try:
        store = ReviewStore(args.data_dir)
        store.validate_audio_paths()
        window = ReviewWindow(store, audio_enabled=not args.smoke_test)
    except Exception as exc:
        if args.smoke_test:
            print(f"A4 GUI startup failed: {exc}", file=sys.stderr); return 1
        QMessageBox.critical(None, "A4 데이터 오류", str(exc)); return 1
    if args.smoke_test:
        app.processEvents()
        if window.clip_list.count() != 24 or store.progress()[1] != 24:
            print("A4 GUI smoke test expected 24 clips", file=sys.stderr); return 1
        print("A4 GUI smoke test: PASS (24 clips)")
        window.close(); return 0
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
