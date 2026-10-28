"""Real Qt model/overlay delivery; no WASAPI, network or screen capture."""
import json
import tempfile
import unittest
from pathlib import Path

from PySide6.QtWidgets import QApplication
from phase5.gui_app import SubtitleFeedModel
from phase5.subtitle_overlay import SubtitleOverlay
from phase5.live_caption_controller import configure_gui_runtime, shutdown_gui_runtime


class QtCaptionTraceTests(unittest.TestCase):
    def test_opt_in_runtime_traces_model_overlay_rejection_and_expiry(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            configure_gui_runtime(directory / 'runtime.log', diagnostic_dir=directory)
            model, overlay = SubtitleFeedModel(), SubtitleOverlay()
            model.entries_changed.connect(overlay.set_entries)
            try:
                partial = dict(speaker='speaker_0', utterance_id=1, sequence=1,
                               status='partial', text='확인할 문장', session_id='qt-session', window_index=0)
                self.assertTrue(model.upsert_event(partial))
                self.assertTrue(model.upsert_event(dict(partial, sequence=2, status='final')))
                self.assertFalse(model.upsert_event(dict(partial, sequence=2, status='final')))
                self.assertEqual(model.rowCount(), 1)
                self.assertEqual(overlay._labels[0].text(), '확인할 문장')
                model._presentation.expire(float('inf'))
                model._refresh_snapshot()
                self.assertEqual(model.rowCount(), 0)
                self.assertEqual(overlay._labels[0].text(), '')
            finally:
                model._expiry_timer.stop()
                overlay.close()
                app.processEvents()
                shutdown_gui_runtime()
            paths = list(directory.glob('gui_trace_*.jsonl'))
            self.assertEqual(len(paths), 1)
            rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
            stages = {row['stage'] for row in rows}
            self.assertTrue({'presentation_decision', 'gui_model_delivery', 'overlay_delivery'} <= stages)
            self.assertTrue(any(row.get('reason') == 'duplicate_event' for row in rows))
            self.assertEqual(rows[-1]['entries'], [])
