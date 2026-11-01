from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
    from a4_review_gui import ReviewWindow, diff_html
    from review_store import ReviewStore
except ImportError:
    QApplication = None


HERE = Path(__file__).resolve().parent


@unittest.skipIf(QApplication is None, "PySide6 is not installed in this environment")
class ReviewGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.temp = Path(self.temporary.name)
        for name in ("review_template.csv", "review_dataset.json", "evaluate_reviewed.py"):
            shutil.copy2(HERE / name, self.temp / name)
        store = ReviewStore(self.temp, project_root=HERE.parents[1], completed_path=self.temp / "review_completed.csv")
        self.window = ReviewWindow(store, audio_enabled=False)
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.temporary.cleanup()

    def test_startup_loads_24_and_filters(self):
        self.assertEqual(self.window.clip_list.count(), 24)
        self.assertEqual(self.window.progress_label.text(), "검수 진행률: 0/24")
        self.window.filter_combo.setCurrentIndex(2)
        self.app.processEvents()
        self.assertEqual(self.window.clip_list.count(), 0)

    def test_copy_does_not_verify(self):
        sample_id = self.window.current_id
        self.window._copy_model("turbo_transcript")
        self.assertEqual(self.window.store.get(sample_id)["review_status"], "pending")
        self.assertTrue(self.window.dirty)

    def test_diff_highlights_changed_text(self):
        left, right = diff_html("안녕하세요", "안녕하세여")
        self.assertIn("background", left)
        self.assertIn("background", right)


if __name__ == "__main__":
    unittest.main()
