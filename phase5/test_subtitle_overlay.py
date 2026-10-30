"""B-2: actual QLabel measurements, with no audio, STT, or network access."""
from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QApplication, QLabel

from phase5.subtitle_overlay import OverlayStyle, SubtitleOverlay
from phase5.subtitle_presentation import SubtitleFeedEntry, SubtitlePresentationState


class OverlayLineBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self) -> None:
        self.overlay = SubtitleOverlay()
        self.overlay.resize(800, self.overlay.height())
        self.overlay.show()
        self.app.processEvents()

    def tearDown(self) -> None:
        self.overlay.close()
        self.app.processEvents()

    @staticmethod
    def entry(text: str, sequence: int = 1, status: str = "partial") -> SubtitleFeedEntry:
        return SubtitleFeedEntry("speaker_0", sequence, sequence, status, text,
                                 timestamp=123, session_id="b2", window_index=4)

    def label_lines(self, label: QLabel, text: str | None = None) -> int:
        # Independent oracle: ask QLabel for its rendered height, calibrated
        # against one and two explicit lines in the same font and paint device.
        probe = QLabel(self.overlay)
        probe.hide()
        probe.setFont(label.font())
        probe.setWordWrap(True)
        probe.setTextFormat(Qt.TextFormat.PlainText)
        probe.setText("M")
        one = probe.heightForWidth(label.width())
        probe.setText("M\nM")
        spacing = probe.heightForWidth(label.width()) - one
        probe.setText(label.text() if text is None else text)
        height = probe.heightForWidth(label.width())
        probe.deleteLater()
        return 1 + max(0, (height - one + spacing - 1) // spacing)

    def check_budget(self, expected: int | None = None) -> list[QLabel]:
        self.app.processEvents()
        labels = [label for label in self.overlay._labels if label.isVisible()]
        count = sum(self.label_lines(label) for label in labels)
        self.assertLessEqual(count, 3)
        if expected is not None:
            self.assertEqual(count, expected)
        self.assertEqual(len(labels), len(self.overlay.entries))
        self.assertLessEqual(self.overlay.height(), self.overlay.layout_metrics.maximum_height)
        for label in labels:
            self.assertLessEqual(label.heightForWidth(label.width()), label.height())
            self.assertTrue(self.overlay.rect().contains(label.mapTo(self.overlay, label.rect().bottomRight())))
        return labels

    def test_one_two_three_explicit_lines_stay_whole(self) -> None:
        for lines in (1, 2, 3):
            with self.subTest(lines=lines):
                entry = self.entry("\n".join(["현재 자막"] * lines))
                self.overlay.set_entries((entry,))
                label = self.check_budget(lines)[0]
                self.assertEqual(label.text(), entry.text)
                self.assertEqual(label.toolTip(), "")

    def test_four_or_more_lines_keep_latest_suffix_and_source(self) -> None:
        for status in ("partial", "final"):
            for lines in (4, 12):
                with self.subTest(status=status, lines=lines):
                    entry = self.entry("이전 문장\n" * (lines - 1) + "최신 내용 123!",
                                       status=status)
                    original = replace(entry)
                    self.overlay.set_entries((entry,))
                    label = self.check_budget(3)[0]
                    self.assertTrue(label.text().startswith("…"))
                    self.assertTrue(label.text().endswith("최신 내용 123!"))
                    self.assertEqual(entry, original)
                    self.assertIs(self.overlay.entries[0], entry)
                    self.assertEqual(label.toolTip(), entry.text)

    def test_final_two_lines_and_partial_one_line(self) -> None:
        final = self.entry("확정 첫 줄\n확정 둘째 줄", status="final")
        partial = self.entry("진행 중", 2)
        self.overlay.set_entries((final, partial))
        self.check_budget(3)
        self.assertEqual(self.overlay.entries, (final, partial))
        self.assertEqual([label.text() for label in self.overlay._labels[:2]],
                         [final.text, partial.text])

    def test_final_two_lines_and_partial_two_lines_keep_newest_whole(self) -> None:
        final = self.entry("이전 첫 줄\n이전 둘째 줄", status="final")
        partial = self.entry("최신 첫 줄\n최신 둘째 줄", 2)
        self.overlay.set_entries((final, partial))
        self.check_budget(2)
        self.assertEqual(self.overlay.entries, (partial,))
        self.assertEqual(self.overlay._source_entries, (final, partial))
        self.assertEqual(self.overlay._labels[0].text(), partial.text)

    def test_partial_three_lines_hides_finals_then_restores_them(self) -> None:
        finals = (self.entry("확정 A", status="final"),
                  self.entry("확정 B", 2, "final"))
        partial = self.entry("최신 1\n최신 2\n최신 3", 3)
        self.overlay.set_entries((*finals, partial))
        self.check_budget(3)
        self.assertEqual(self.overlay.entries, (partial,))
        self.overlay.set_entries(finals)
        self.check_budget(2)
        self.assertEqual(self.overlay.entries, finals)

    def test_final_expiry_reflows_without_changing_deadlines(self) -> None:
        now = [0.0]
        state = SubtitlePresentationState(clock=lambda: now[0])
        state.apply(dict(speaker="speaker_0", utterance_id=1, sequence=1,
                         status="final", text="확정 첫 줄\n확정 둘째 줄"))
        state.apply(dict(speaker="speaker_0", utterance_id=2, sequence=2,
                         status="partial", text="최신 자막"))
        snapshot = state.entries
        self.overlay.set_entries(snapshot)
        self.check_budget(3)
        self.assertFalse(state.expire(3.999))
        self.assertEqual(state.entries, snapshot)
        self.assertTrue(state.expire(4.0))
        self.overlay.set_entries(state.entries)
        self.check_budget(1)
        self.assertEqual(self.overlay.entries, (snapshot[-1],))

    def test_repeated_partial_updates_do_not_mutate_feed_or_create_rows(self) -> None:
        state = SubtitlePresentationState()
        for sequence in range(1, 16):
            text = "과거 문장입니다. " * (10 + sequence) + f"최신 {sequence:02d}!"
            state.apply(dict(speaker="speaker_0", utterance_id=1, sequence=sequence,
                             status="partial", text=text))
            self.overlay.set_entries(state.entries)
            label = self.check_budget()[0]
            self.assertEqual(len(self.overlay.entries), 1)
            self.assertEqual(state.entries[0].text, text)
            self.assertEqual(self.overlay.entries, state.entries)
            self.assertTrue(label.text().startswith("…"))
            self.assertTrue(label.text().endswith(f"최신 {sequence:02d}!"))

    def test_hidden_final_reappears_only_until_its_original_expiry(self) -> None:
        state = SubtitlePresentationState(clock=lambda: 0.0)
        state.apply(dict(speaker="speaker_0", utterance_id=1, sequence=1,
                         status="final", text="원본 FINAL"))
        final = state.entries[0]
        partial = dict(speaker="speaker_0", utterance_id=2, sequence=2,
                       status="partial", text="첫 줄\n둘째 줄\n마지막 줄")
        state.apply(partial)
        self.overlay.set_entries(state.entries)
        self.check_budget(3)
        self.assertNotIn(final, self.overlay.entries)
        self.assertIn(final, state.entries)
        state.apply(dict(partial, sequence=3, text="짧아진 PARTIAL"))
        self.assertFalse(state.expire(3.999))
        self.overlay.set_entries(state.entries)
        self.check_budget(2)
        self.assertIs(self.overlay.entries[0], final)
        self.assertTrue(state.expire(4.0))
        self.overlay.set_entries(state.entries)
        self.check_budget(1)
        self.assertNotIn(final, self.overlay.entries)

    def test_automatic_wrap_uses_qt_measurements(self) -> None:
        self.overlay.set_entries((self.entry("준비"),))
        self.app.processEvents()
        label = self.overlay._labels[0]
        for word in ("한국어 자막 ", "Ab12, "):
            for target in (1, 2, 3, 4):
                with self.subTest(word=word, target=target):
                    text = word
                    while self.label_lines(label, text) < target:
                        text += word
                    self.assertEqual(self.label_lines(label, text), target)
                    entry = self.entry(text.rstrip())
                    self.overlay.set_entries((entry,))
                    shown = self.check_budget(min(target, 3))[0].text()
                    if target <= 3:
                        self.assertEqual(shown, entry.text)
                    else:
                        self.assertTrue(shown.startswith("…"))
                        self.assertTrue(entry.text.endswith(shown[1:]))

    def test_combining_mark_is_not_separated_at_suffix_boundary(self) -> None:
        cluster = "e\u0301\u0327"
        entry = self.entry((cluster + " ") * 160)
        self.overlay.set_entries((entry,))
        shown = self.check_budget()[0].text()
        self.assertTrue(shown.startswith("…e\u0301\u0327"))
        self.assertTrue(entry.text.rstrip().endswith(shown[1:]))
        self.assertEqual(self.overlay.entries[0].text, entry.text)

    def test_newlines_blank_lines_and_markup_are_literal(self) -> None:
        for text in ("첫 줄\n\n끝 줄", "<b>literal</b>\n42 & 3 < 9\n마지막"):
            with self.subTest(text=text):
                self.overlay.set_entries((self.entry(text),))
                self.assertEqual(self.check_budget(3)[0].text(), text)

    def test_width_and_font_changes_reflow_existing_entries(self) -> None:
        entry = self.entry("폰트 크기와 화면 폭에 따라 자동 줄바꿈됩니다. " * 12 + "최신 끝")
        self.overlay.set_entries((entry,))
        for size in (16, 28, 52):
            for width in (320, 800, 1200):
                with self.subTest(size=size, width=width):
                    style = OverlayStyle(size, 55, False, "#A7F3D0")
                    self.overlay.apply_style(style)
                    self.overlay.resize(width, self.overlay.height())
                    label = self.check_budget()[0]
                    self.assertEqual(self.overlay.style_config, style)
                    self.assertEqual(label.font().pointSize(), size)
                    self.assertTrue(label.text().endswith("끝"))
                    if not label.text().endswith("최신 끝"):
                        # At high scale + 52pt + 320px, this complete phrase
                        # can need two lines while the screen cap allows one.
                        # Require QLabel evidence that the longer suffix cannot fit.
                        probe = QLabel(self.overlay)
                        probe.hide()
                        probe.setWordWrap(True)
                        probe.setFont(label.font())
                        probe.setText("…최신 끝")
                        self.assertGreater(
                            probe.heightForWidth(label.width()),
                            self.overlay.layout_metrics.maximum_height
                            - self.overlay._non_text_height(1),
                        )
                        probe.deleteLater()
                    self.assertEqual(self.overlay.entries, (entry,))

    def test_narrowing_hides_and_widening_restores_older_final(self) -> None:
        self.overlay.apply_style(OverlayStyle(font_size=16))
        final = self.entry("이전 확정", status="final")
        partial = self.entry("화면 폭에 맞춰 최신 발화를 표시합니다. " * 3, 2)
        self.overlay.set_entries((final, partial))
        self.overlay.resize(400, self.overlay.height())
        self.check_budget()
        self.assertEqual(self.overlay.entries, (partial,))
        self.overlay.resize(1600, self.overlay.height())
        self.check_budget()
        self.assertEqual(self.overlay.entries, (final, partial))

    def test_small_work_area_limits_height_even_below_three_lines(self) -> None:
        screen = Mock()
        screen.availableGeometry.return_value = QRect(0, 0, 640, 240)
        with patch("phase5.subtitle_overlay.QGuiApplication.primaryScreen", return_value=screen):
            self.overlay.apply_style(OverlayStyle(font_size=52))
            self.overlay.resize(500, self.overlay.height())
            self.overlay.set_entries((self.entry("오래된 발화\n" * 20 + "최신"),))
            label = self.check_budget(1)[0]
            self.assertTrue(label.text().endswith("최신"))

    def test_short_final_and_partial_keep_original_text_and_status(self) -> None:
        for status in ("final", "partial"):
            with self.subTest(status=status):
                entry = self.entry("짧은 자막 123!", status=status)
                self.overlay.set_entries((entry,))
                label = self.check_budget(1)[0]
                self.assertEqual(label.text(), entry.text)
                self.assertIs(self.overlay.entries[0], entry)
                self.assertEqual(self.overlay.entries[0].status, status)


if __name__ == "__main__":
    unittest.main()
