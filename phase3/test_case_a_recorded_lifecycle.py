"""Characterize recorded SINGLE behavior; these tests do not judge STT accuracy."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from test_general_subtitle_lifecycle import GeneralSubtitleHarness


FIXTURE = Path(__file__).parent / "test_fixtures" / "case_a_20261031_windows.json"
QUESTION = "뭐가 바퀴었어?"


class RecordedCaseALifecycleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.harness = GeneralSubtitleHarness()
        self.replayed = {}
        self.sequence = 0
        for row in self.fixture["windows"]:
            evidence = row["audio_overlap_evidence"]
            assembly, events = self.harness.speech(
                row["window"], row["raw"],
                source_supported=row["source_supported"],
                shared_speech=evidence["shared_speech"],
                supported_tail_revision=evidence["supported_tail_revision"],
            )
            public_events = []
            for event in events:
                if event.publication_text:
                    self.sequence += 1
                    public_events.append({
                        "sequence": self.sequence,
                        "utterance_id": event.utterance_id,
                        "status": event.status,
                        "action": event.action,
                        "text": event.publication_text,
                        "finalize_reason": event.finalize_reason,
                    })
            self.replayed[row["window"]] = assembly, public_events

    def test_recorded_assembly_and_public_events_match_including_warmup(self):
        self.assertEqual([r["window"] for r in self.fixture["windows"]], list(range(7, 19)))
        for row in self.fixture["windows"]:
            with self.subTest(window=row["window"]):
                assembly, events = self.replayed[row["window"]]
                for field, expected in row["expected_assembly"].items():
                    self.assertEqual(getattr(assembly, field), expected, field)
                self.assertEqual(events, row["expected_events"])
        self.assertEqual(self.sequence, 17)

    def test_w13_preserves_multi_sentence_raw_in_one_partial(self):
        assembly, events = self.replayed[13]
        self.assertEqual(assembly.raw, QUESTION + " 인공지능은")
        self.assertEqual(assembly.utterance_hypothesis, assembly.raw)
        self.assertEqual([(e["sequence"], e["status"], e["utterance_id"]) for e in events],
                         [(10, "final", 4), (11, "partial", 5)])
        self.assertEqual(events[-1]["text"], assembly.raw)

    def test_w14_fuzzy_replacement_preserves_question(self):
        assembly, events = self.replayed[14]
        self.assertEqual(assembly.match_type, "fuzzy_replace")
        self.assertEqual(assembly.rollover_text, "")
        self.assertEqual(events[0]["text"],
                         QUESTION + " 인공지능 음성 인식 시스템은 오디오 대신")
        self.assertEqual(events[0]["utterance_id"], 5)
        self.assertEqual(events[0]["action"], "replace")

    def test_w15_extends_the_existing_partial(self):
        before = self.replayed[14][0].utterance_hypothesis
        assembly, events = self.replayed[15]
        self.assertEqual(assembly.utterance_hypothesis, before + " " + assembly.raw)
        self.assertEqual(events[0]["action"], "extend")
        self.assertEqual(events[0]["utterance_id"], 5)
        self.assertTrue(events[0]["text"].endswith("텍스트로 변화합니다."))

    def test_w16_corrects_the_ending_without_premature_final(self):
        before = self.replayed[15][0].utterance_hypothesis
        assembly, events = self.replayed[16]
        self.assertEqual(assembly.match_type, "fuzzy_replace")
        self.assertEqual(assembly.rollover_text, "")
        self.assertEqual(assembly.utterance_hypothesis,
                         before.removesuffix("변화합니다.") + "변환합니다.")
        self.assertEqual([(e["sequence"], e["status"], e["utterance_id"]) for e in events],
                         [(14, "partial", 5)])

    def test_w17_finalizes_once_before_new_partial_and_w18_continues(self):
        assembly, events = self.replayed[17]
        expected_final = self.replayed[16][0].utterance_hypothesis
        self.assertEqual(assembly.rollover_text, expected_final)
        self.assertEqual(assembly.rollover_reason, "sentence_boundary_confirmed")
        self.assertEqual([(e["sequence"], e["status"], e["utterance_id"]) for e in events],
                         [(15, "final", 5), (16, "partial", 6)])
        self.assertEqual(events[0]["text"], expected_final)
        self.assertEqual(events[1]["text"], "비상청은 오늘 오후 새로운 안내를")
        self.assertEqual(self.harness.state.final_segments.count(expected_final), 1)
        following = self.replayed[18][1]
        self.assertEqual(following[0]["utterance_id"], 6)
        self.assertEqual(following[0]["text"],
                         "비상청은 오늘 오후 새로운 안내를 발표했습니다.")
        self.assertNotIn(QUESTION, following[0]["text"])


if __name__ == "__main__":
    unittest.main()
