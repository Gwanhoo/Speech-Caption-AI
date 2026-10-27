"""P0-2: synthetic SINGLE path; no Windows device or Whisper download."""
from __future__ import annotations

import base64
import unittest
from unittest.mock import Mock, patch

import numpy as np

import gpu_processing_server as server
from remote_gpu_client import parse_result
from remote_gpu_protocol import decode_binary_result, encode_binary_result
from stt_context import SttResult, transcribe_base
from test_remote_gpu_client import payload
import test_remote_overlap_pipeline as worker_tests
from phase5.subtitle_presentation import SubtitlePresentationState


class GhostCaptionPathTests(unittest.TestCase):
    def test_server_vad_gate_and_stt_diagnostics_roundtrip(self):
        models = server.LoadedModels(None, None, Mock(), Mock())
        models.whisper.model.device = "cuda"
        models.whisper.model.compute_type = "float16"
        audio = np.full(48000, .01, dtype=np.float32)
        vad = dict(speech_detected=False, speech_duration_ms=0., speech_ratio=0.,
                   rms=.01, peak=.01, processing_seconds=0., timestamps=[],
                   audio_duration_ms=3000.)
        for text in ("네", "감사", "감사합니다", "다음 소식입니다"):
            with self.subTest(text=text), \
                    patch.object(server.PipelineService, "_load_models", return_value=models), \
                    patch.object(server, "detect_speech_activity", return_value=vad), \
                    patch.object(server, "transcribe_base", return_value=SttResult(
                        text, .01, 1, -.2, .05, 1.2)) as transcribe:
                service = server.PipelineService(warm_up=False)
                try:
                    result = service.process("silence", audio, 3.)
                    transcribe.assert_not_called()
                    self.assertEqual(result["speakers"][0]["raw_transcript"], "")
                    self.assertEqual(result["speakers"][0]["stt_diagnostics"]["skip_reason"], "vad_inactive")
                    with patch.object(server, "detect_speech_activity", return_value=dict(
                            vad, speech_detected=True, speech_duration_ms=300.,
                            speech_ratio=.1, timestamps=[{"start": 0, "end": 4800}])):
                        result = service.process("speech", audio, 3.)
                    transcribe.assert_called_once()
                    result["window_index"] = 0
                    parsed = parse_result(decode_binary_result(encode_binary_result(result)),
                                          "speech", 0, 48000, .02)
                    self.assertEqual(parsed.slots[0]["raw_transcript"], text)
                    self.assertEqual(parsed.slots[0]["stt_diagnostics"], dict(
                        executed=True, skip_reason=None, segment_count=1,
                        average_log_probability=-.2, maximum_no_speech_probability=.05,
                        maximum_compression_ratio=1.2))
                finally:
                    service.close()

    def test_single_repeated_short_speech_silence_and_expiry(self):
        def response(audio, index):
            result = payload("r", index, len(audio))
            for i, slot in enumerate(result["speakers"]):
                active = i == 0 and index in (0, 2)
                wave = audio if active else np.zeros_like(audio)
                slot["waveform"]["data"] = base64.b64encode(wave.astype("<f4").tobytes()).decode()
                slot["raw_transcript"] = "네" if active else ""
                slot["vad"].update(speech_detected=active,
                    speech_duration_ms=3000. if active else 0., speech_ratio=1. if active else 0.,
                    timestamps=[{"start": 0, "end": len(audio)}] if active else [])
                slot["stt_diagnostics"] = {"executed": active}
            return result

        harness = worker_tests.PipelineTests()
        code, result = harness.run_pipeline(True, response_factory=response, duration=9,
                                           speaker_mode="single", websocket=False, evidence=True)
        self.assertEqual(code, 0)
        self.assertEqual(result["errors"], [])
        rows = result["windows"]
        self.assertEqual([(r["stream_start_seconds"], r["stream_end_seconds"]) for r in rows],
                         [(0, 3), (2, 5), (4, 7), (6, 9)])
        for row in rows:
            self.assertEqual(row["session_id"], result["session_id"])
            self.assertEqual(row["remote_request_id"], "r")
            self.assertIsInstance(row["remote_stt_diagnostics"][0]["executed"], bool)
        for record in harness.last_evidence_records:
            if record["record_type"] == "window":
                self.assertEqual(record["session_id"], result["session_id"])
                self.assertEqual(record["remote_request_id"], "r")
                self.assertIsNotNone(record["remote_stt_diagnostics"])
        events = result["subtitle_events"]
        self.assertEqual([(e["status"], e["text"]) for e in events],
                         [("partial", "네"), ("final", "네"), ("partial", "네"), ("final", "네")])
        self.assertNotEqual(events[0]["utterance_id"], events[2]["utterance_id"])
        state = SubtitlePresentationState(clock=lambda: 0.)
        with self.assertLogs("voxlux.presentation", level="INFO") as logs:
            for event in events:
                self.assertEqual(event["session_id"], result["session_id"])
                state.apply(event)
            self.assertTrue(state.expire(now=4.))
        self.assertEqual(state.entries, ())
        self.assertIn("session_id=" + repr(result["session_id"]), "\n".join(logs.output))
        self.assertIn("window_index=3", "\n".join(logs.output))

    def test_inconsistent_response_is_dropped_before_assembler_and_next_speech_survives(self):
        def response(audio, index):
            result = payload("r", index, len(audio))
            for i, slot in enumerate(result["speakers"]):
                active = i == 0 and index == 1
                slot["waveform"]["data"] = base64.b64encode(
                    (audio if active else audio * 0).astype("<f4").tobytes()).decode()
                slot["raw_transcript"] = ("네" if active else "임의의 허위 문장" if i == 0 else "")
                slot["vad"].update(speech_detected=active,
                    speech_duration_ms=3000. if active else 0., speech_ratio=1. if active else 0.,
                    timestamps=[{"start": 0, "end": len(audio)}] if active else [])
            return result

        harness = worker_tests.PipelineTests()
        code, result = harness.run_pipeline(True, response_factory=response,
                                           speaker_mode="single", websocket=False)
        self.assertEqual(code, 1)  # Deliberately malformed first response is an error.
        self.assertTrue(result["errors"])
        self.assertIn("inactive VAD", str(result["errors"]))
        self.assertEqual([r["window"] for r in result["windows"]], [1])
        self.assertTrue(result["subtitle_events"])
        self.assertEqual({e["text"] for e in result["subtitle_events"]}, {"네"})

    def test_transcribe_empty_and_nonfinite_diagnostics_do_not_create_text(self):
        model = Mock()
        model.transcribe.return_value = (iter([]), None)
        self.assertEqual(transcribe_base(model, np.zeros(48000, np.float32)).text, "")
        model.transcribe.return_value = (iter([Mock(text="네", avg_logprob=float("nan"),
            no_speech_prob=float("nan"), compression_ratio=float("nan"))]), None)
        result = transcribe_base(model, np.ones(48000, np.float32))
        self.assertEqual(result.text, "네")
        self.assertIsNone(result.average_log_probability)
        self.assertIsNone(result.maximum_no_speech_probability)
        self.assertIsNone(result.maximum_compression_ratio)


if __name__ == "__main__":
    unittest.main()
