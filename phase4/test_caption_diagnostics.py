import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from caption_diagnostics import LOGGER, start_trace, stop_trace, trace_event
from remote_gpu_client import RemoteGPUClient, RemoteTimeoutError
from phase5.subtitle_presentation import SubtitlePresentationState


class CaptionDiagnosticTests(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertFalse(LOGGER.handlers)
        with patch.object(LOGGER, "info") as output:
            trace_event("raw_stt", text="기본 비활성")
        output.assert_not_called()

    def test_presentation_accept_replay_and_expiry_are_correlated(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            handler = start_trace(path)
            try:
                state = SubtitlePresentationState(clock=lambda: 10)
                event = dict(speaker="speaker_0", utterance_id=1, sequence=1,
                             status="partial", text="실제 문장", session_id="session-a", window_index=2)
                self.assertTrue(state.apply(event))
                self.assertFalse(state.apply(event))
                self.assertTrue(state.apply(dict(event, sequence=2, status="final")))
                self.assertTrue(state.expire(14))
            finally:
                stop_trace(handler)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r["reason"] for r in rows],
                             ["event_applied", "duplicate_event", "event_applied", "final_lifetime_elapsed"])
            self.assertTrue(all(r["event"]["session_id"] == "session-a" for r in rows))
            self.assertEqual(rows[-1]["entries"], [])
            self.assertTrue(all(r["wall_time_ns"] > 0 and r["monotonic_ns"] > 0 for r in rows))
            with self.assertRaises(FileExistsError):
                start_trace(path)

    def test_timeout_trace_omits_url_headers_and_exception_message(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            handler = start_trace(path)
            client = RemoteGPUClient("https://user:credential@example.invalid?token=secret")
            client.diagnostic_session_id = "session-b"
            try:
                with patch.object(client, "_request_with_timing", side_effect=RemoteTimeoutError("secret")):
                    with self.assertRaises(RemoteTimeoutError):
                        client.process(np.zeros(160, dtype=np.float32), 3)
            finally:
                client.close()
                stop_trace(handler)
            content = path.read_text()
            for secret in ("credential", "secret", "example.invalid", "Authorization"):
                self.assertNotIn(secret, content)
            rows = [json.loads(line) for line in content.splitlines()]
            self.assertEqual([r["stage"] for r in rows], ["request_sent", "request_error"])
            self.assertEqual(rows[0]["request_id"], rows[1]["request_id"])
            self.assertEqual(rows[0]["session_id"], "session-b")

    def test_disk_failure_does_not_break_caption_delivery(self):
        with tempfile.TemporaryDirectory() as folder:
            handler = start_trace(Path(folder) / "trace.jsonl")
            try:
                with patch.object(handler.stream, "write", side_effect=OSError("disk full")):
                    with self.assertLogs("voxlux", logging.WARNING):
                        trace_event("subtitle_dispatch", text="보존할 자막")
                    trace_event("subtitle_dispatch", text="다음 자막")
                self.assertTrue(handler.failed)
            finally:
                stop_trace(handler)
