"""Headless tests for the standalone UI shell; no WASAPI or server calls."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from phase5.ui_shell import CaptionStyle, MainWindow, SettingsDialog, SubtitleOverlay


class GuiShellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def tearDown(self) -> None:
        for widget in self.app.topLevelWidgets():
            widget.close()

    def test_initial_state(self) -> None:
        window = MainWindow()
        self.assertIsNone(window.overlay)
        self.assertFalse(window.start_button.isEnabled())
        self.assertFalse(window.stop_button.isEnabled())

    def test_create_start_stop_state_transition_and_overlay_persistence(self) -> None:
        window = MainWindow()
        window.create_overlay()
        self.assertIsInstance(window.overlay, SubtitleOverlay)
        self.assertTrue(window.start_button.isEnabled())
        self.assertFalse(window.stop_button.isEnabled())

        window.start_caption()
        self.assertFalse(window.start_button.isEnabled())
        self.assertTrue(window.stop_button.isEnabled())
        self.assertIn("오늘 저녁", window.overlay.caption.text())

        overlay = window.overlay
        window.stop_caption()
        self.assertIs(window.overlay, overlay)
        self.assertTrue(window.start_button.isEnabled())
        self.assertFalse(window.stop_button.isEnabled())

    def test_empty_subtitle_is_ignored(self) -> None:
        window = MainWindow()
        window.create_overlay()
        previous = window.overlay.caption.text()
        window.update_subtitle("")
        self.assertEqual(window.overlay.caption.text(), previous)

    def test_settings_path_is_safe_before_overlay_creation(self) -> None:
        window = MainWindow()
        dialog = SettingsDialog(CaptionStyle(), window)
        dialog.close()
        self.assertIsNone(window.overlay)


if __name__ == "__main__":
    unittest.main()
