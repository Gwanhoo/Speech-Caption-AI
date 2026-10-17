"""Regression tests for the development-only production-overlay preview."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from phase5.run_overlay_preview import OverlayPreviewSession, populate_preview
from phase5.subtitle_overlay import OverlayStyle
from phase5.subtitle_presentation import SubtitlePresentationState


class OverlayPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def test_standard_preview_contains_two_finals_and_one_partial(self) -> None:
        state = SubtitlePresentationState()
        populate_preview(state, "standard")

        self.assertEqual(len(state.entries), 3)
        self.assertEqual([entry.status for entry in state.entries], ["final", "final", "partial"])
        self.assertTrue(all(len(entry.text) > 30 for entry in state.entries))

    def test_overflow_preview_supplies_three_long_sentences(self) -> None:
        state = SubtitlePresentationState()
        populate_preview(state, "overflow")

        self.assertEqual(len(state.entries), 3)
        self.assertEqual([entry.status for entry in state.entries], ["final", "final", "partial"])
        self.assertTrue(all(len(entry.text) > 200 for entry in state.entries))

    def test_preview_session_applies_style_to_the_production_overlay(self) -> None:
        session = OverlayPreviewSession(
            "standard",
            OverlayStyle(font_size=38, background_opacity=60, bold=False, text_color="#A7F3D0"),
        )
        try:
            self.assertEqual(session.overlay.style_config.font_size, 38)
            self.assertEqual(session.overlay.style_config.background_opacity, 60)
            self.assertFalse(session.overlay.style_config.bold)
            self.assertEqual(session.overlay.style_config.text_color, "#A7F3D0")
            # The production overlay may trim older wrapped rows to fit the
            # available display.  Its snapshot must nevertheless be drawn
            # from the presentation-state entries and keep the newest partial.
            self.assertTrue(session.overlay.entries)
            self.assertTrue(all(entry in session.state.entries for entry in session.overlay.entries))
            self.assertEqual(session.overlay.entries[-1].status, "partial")
        finally:
            session.close()

    def test_overflow_preview_is_capped_and_keeps_the_newest_entry(self) -> None:
        session = OverlayPreviewSession(
            "overflow",
            OverlayStyle(font_size=40, background_opacity=72, bold=True, text_color="#FFFFFF"),
        )
        try:
            metrics = session.overlay.layout_metrics
            self.assertLessEqual(metrics.window_height, metrics.maximum_height)
            self.assertLess(metrics.visible_rows, metrics.source_rows)
            self.assertEqual(session.overlay.entries[-1], session.state.entries[-1])
            self.assertEqual(len(session.state.entries), 3)
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
