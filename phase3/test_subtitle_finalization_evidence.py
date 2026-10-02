"""Evidence/lifecycle tests; text is supplied, never inferred by a GPU model."""
import unittest
from unittest.mock import patch

import numpy as np

from secondary_leakage_diagnostics import admit_subtitle_streams
from subtitle_assembler import SpeakerSubtitleState, SubtitleAssembler


class FinalizationEvidenceTests(unittest.TestCase):
    def test_unsupported_tentative_is_discarded_on_silence_and_flush(self):
        for flush in (False, True):
            state = SpeakerSubtitleState(0, require_final_support=True)
            partial = state.process(0, "미확인 발화", True, 3,
                confirmed_prefix_length=0, source_supported=False)[0]
            self.assertEqual(partial.stable_text, "")
            event = state.flush(0, 3) if flush else state.process(1, "", False, 5)[0]
            self.assertEqual(event.action, "discard")
            self.assertEqual(event.text, "")
            self.assertEqual(state.final_segments, [])
            self.assertEqual(state.partial_text, "")
            self.assertIsNone(state.flush(1, 5))
            next_event = state.process(2, "네", True, 7, source_supported=True)[0]
            self.assertEqual(next_event.utterance_id, 2)
            self.assertEqual(state.flush(2, 7).text, "네")

    def test_source_supported_short_speech_and_temporal_prefix_are_preserved(self):
        state = SpeakerSubtitleState(0, require_final_support=True)
        state.process(0, "네", True, 3, confirmed_prefix_length=0, source_supported=True)
        self.assertEqual(state.process(1, "", False, 5)[0].text, "네")
        state.process(2, "확인된 내용", True, 7, confirmed_prefix_length=0, source_supported=False)
        state.process(3, "확인된 내용 미확인 꼬리", True, 9,
                      confirmed_prefix_length=5, source_supported=False)
        final = state.flush(3, 9)
        self.assertEqual(final.text, "확인된 내용")
        self.assertNotIn("미확인", state.final_segments[-1])

    def test_copied_unsupported_prefix_does_not_gain_later_source_support(self):
        state = SpeakerSubtitleState(0, require_final_support=True)
        state.process(0, "미확인 후보", True, 3, confirmed_prefix_length=0,
                      source_supported=False, source_text="미확인 후보")
        state.process(1, "미확인 후보 새로운 발화", True, 5, confirmed_prefix_length=0,
                      source_supported=True, source_text="새로운 발화")
        self.assertEqual(state.flush(1, 5).text, "")

    def test_discard_does_not_commit_assembler_session_history(self):
        assembler = SubtitleAssembler(0)
        assembler.process(0, "미확인 후보")
        assembler.reset_utterance(final_text="")
        next_event = assembler.process(2, "정상 발화")
        self.assertEqual(next_event.session_text, "정상 발화")

    def test_live_ambiguous_admission_remains_partial_but_not_final(self):
        audio = np.random.default_rng(17).normal(size=4800).astype(np.float32)
        vad = {"speech_detected": True, "speech_duration_ms": 764, "speech_ratio": .255,
               "timestamps": [{"start": 0, "end": 1200}]}
        for direct, owner, independent in ((.1044, .9847, .2842), (.4382, .8969, .30)):
            # First tuple is recorded evidence. Second uses a declared synthetic
            # independent correlation because that value was absent in the log.
            evidence = {"candidate_input_correlation": direct, "owner_input_correlation": owner,
                        "independent_input_correlation": independent, "pair_correlation": .05}
            with patch("secondary_leakage_diagnostics.source_input_evidence", return_value=evidence):
                texts, effective, diagnostic = admit_subtitle_streams(
                    mixture=audio, speakers=(audio, audio), transcripts=["", "임의의 후보"],
                    vad_results=[dict(vad, speech_detected=False, timestamps=[]), vad],
                    active_hypotheses=["", ""], speaker_assignment={})
            self.assertFalse(diagnostic["applied"])
            self.assertFalse(effective[1]["source_supported"])
            state = SpeakerSubtitleState(1, require_final_support=True)
            state.process(0, texts[1], True, 3, confirmed_prefix_length=0,
                          source_supported=effective[1]["source_supported"])
            self.assertEqual(state.flush(0, 3).text, "")


if __name__ == "__main__":
    unittest.main()
