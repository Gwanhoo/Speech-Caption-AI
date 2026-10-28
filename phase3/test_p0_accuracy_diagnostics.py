"""Deterministic CPU reproductions; these do not establish live STT accuracy."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_general_subtitle_lifecycle import GeneralSubtitleHarness


class FinalizedOverlapTests(unittest.TestCase):
    def test_finalized_duplicate_tail_is_not_published_again(self):
        harness = GeneralSubtitleHarness()
        text = "시청해 주셔서 감사합니다."
        harness.speech(0, text)
        _, events = harness.speech(1, text, shared_speech=True)
        self.assertEqual([e.status for e in events], ["final"])
        assembly, events = harness.speech(2, text, shared_speech=True)
        self.assertEqual(events, [])
        self.assertEqual(assembly.utterance_hypothesis, "")
        self.assertEqual(harness.assembler.assembled, text)

    def test_finalized_tail_is_trimmed_before_new_content(self):
        harness = GeneralSubtitleHarness()
        text, new = "시청해 주셔서 감사합니다.", "다음 소식을 전합니다."
        harness.speech(0, text)
        harness.speech(1, text, shared_speech=True)
        _, events = harness.speech(2, text + " " + new, shared_speech=True)
        self.assertEqual(events[-1].publication_text, new)
        self.assertEqual(harness.state.final_segments, [text])

    def test_repetition_without_adjacent_shared_audio_is_preserved(self):
        for window, shared in ((2, False), (2, None), (3, True)):
            with self.subTest(window=window, shared=shared):
                harness = GeneralSubtitleHarness()
                text = "시청해 주셔서 감사합니다."
                harness.speech(0, text)
                harness.speech(1, text, shared_speech=True)
                _, events = harness.speech(window, text, shared_speech=shared)
                self.assertEqual(events[-1].publication_text, text)
                self.assertEqual(events[-1].utterance_id, 2)

    def test_changed_words_after_final_are_not_fuzzy_deleted(self):
        harness = GeneralSubtitleHarness()
        text = "시청해 주셔서 감사합니다."
        harness.speech(0, text)
        harness.speech(1, text, shared_speech=True)
        new = "신청해 주셔서 감사합니다."
        _, events = harness.speech(2, new, shared_speech=True)
        self.assertEqual(events[-1].publication_text, new)


class EvidencePreservationTests(unittest.TestCase):
    def test_reservation_protects_window_wavs_and_crashed_sessions(self):
        from run_overlap_pipeline import reserve_evidence_directory
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            reserve_evidence_directory(output)
            with self.assertRaises(FileExistsError):
                reserve_evidence_directory(output)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "windows").mkdir()
            audio = output / "windows" / "window_000_mixed.wav"
            audio.write_bytes(b"old recording")
            with self.assertRaises(FileExistsError):
                reserve_evidence_directory(output)
            self.assertEqual(audio.read_bytes(), b"old recording")

    def test_existing_evidence_is_never_overwritten(self):
        from run_overlap_pipeline import write_evidence_bundle
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            original, timeline = output / "original.wav", output / "timeline.jsonl"
            original.write_bytes(b"existing audio")
            timeline.write_text("existing timeline", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_evidence_bundle(output, [np.zeros(160, dtype=np.float32)], [], [],
                                      window_seconds=3, stride_seconds=2,
                                      processing_mode="remote", speaker_mode="single")
            self.assertEqual(original.read_bytes(), b"existing audio")
            self.assertEqual(timeline.read_text(encoding="utf-8"), "existing timeline")
