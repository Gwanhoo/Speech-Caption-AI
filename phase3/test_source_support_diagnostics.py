"""Synthetic evidence probes, not recordings or claims about real STT accuracy.

Expected failures document unresolved contracts. Do not turn them into passing
tests by weakening assertions or interpreting correlation as speech identity.
"""
import unittest
from unittest.mock import patch

import numpy as np

from secondary_leakage_diagnostics import admit_subtitle_streams
from speaker_tracking import source_input_evidence
from subtitle_assembler import SpeakerSubtitleState, SubtitleAssembler


class SourceSupportDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.samples = 48000
        t = np.arange(self.samples)
        self.primary = .1 * np.sin(2 * np.pi * 100 * t / self.samples)
        self.other = .1 * np.cos(2 * np.pi * 137 * t / self.samples)

    def vad(self, duration_ms):
        return {
            "speech_detected": bool(duration_ms),
            "speech_duration_ms": duration_ms,
            "speech_ratio": duration_ms / 3000,
            "timestamps": [{"start": 0, "end": duration_ms * 16}] if duration_ms else [],
        }

    def admit(self, mixture, speakers, *, vads=None, texts=None, active=None):
        return admit_subtitle_streams(
            mixture=mixture, speakers=speakers,
            transcripts=texts or ["실제 본문", "분리 후보"],
            vad_results=vads or [self.vad(3000), self.vad(732)],
            active_hypotheses=["", ""] if active is None else active,
            speaker_assignment={"raw_to_logical_mapping": {"0": 0, "1": 1},
                                "active_raw_slots": [0]},
        )

    def partial(self, result, speaker):
        texts, vads, _ = result
        assembly = SubtitleAssembler(speaker).process(0, texts[speaker])
        return SpeakerSubtitleState(speaker, require_final_support=True).process(
            0, assembly.utterance_hypothesis, vads[speaker]["speech_detected"], 3,
            confirmed_prefix_length=assembly.confirmed_prefix_length,
            source_supported=vads[speaker].get("source_supported", False),
            source_text=texts[speaker],
        )

    def low_noise(self, rms):
        noise = np.random.default_rng(17).normal(0, rms, self.samples)
        return self.admit(noise, (noise, noise * 0),
                          vads=[self.vad(316), self.vad(0)],
                          texts=["노이즈에서 생성된 가설", ""])

    def test_missing_owner_intervals_do_not_mean_missing_candidate_support(self):
        result = self.low_noise(.001518)  # Declared synthetic input RMS, not Live RMS.
        decision = result[2]["streams"][0]
        self.assertEqual(decision["reason"], "missing_speech_intervals")
        self.assertTrue(decision["speech_intervals"])
        self.assertEqual(decision["owner_speech_intervals"], [])
        check = decision["source_support_checks"][0]
        self.assertTrue(check["used_direct_fallback"])
        self.assertAlmostEqual(check["direct_correlation"], 1)
        self.assertTrue(check["direct_pass"])
        weak = decision["weak_speech_evidence"]
        self.assertTrue(weak["weak_vad"])
        self.assertTrue(weak["no_localized_burst"])
        self.assertFalse(weak["rms_within_noise_bound"])
        self.assertTrue(self.partial(result, 0)[0].publication_text)

    @unittest.expectedFailure
    def test_unresolved_noise_above_existing_bound_must_not_publish(self):
        # Correlated non-speech exceeds the existing joint-noise guard. No new
        # threshold is justified by the supplied (separated-output) Live RMS.
        events = self.partial(self.low_noise(.001518), 0)
        self.assertFalse(any(event.publication_text for event in events))

    def test_bounded_unlocalized_noise_is_suppressed_before_publication(self):
        result = self.low_noise(.0001)
        self.assertTrue(result[2]["streams"][0]["weak_speech_evidence"]["suppressed"])
        self.assertEqual(self.partial(result, 0), [])

    def test_dominant_primary_with_unrelated_residual_does_not_publish(self):
        result = self.admit(self.primary, (self.primary, self.other))
        self.assertFalse(result[1][1]["source_supported"])
        self.assertTrue(result[2]["streams"][1]["suppressed"])
        self.assertEqual(self.partial(result, 1), [])
        self.assertEqual(self.partial(result, 0)[0].publication_text, "실제 본문")

    def projection_artifact(self):
        # Input contains only p. Separator error introduces r into BOTH outputs.
        return self.admit(self.primary, (self.primary + .01 * self.other, self.other))

    def test_projection_error_can_look_like_an_independent_source(self):
        evidence = source_input_evidence(
            self.primary, self.primary + .01 * self.other, self.other)
        self.assertAlmostEqual(evidence["independent_input_correlation"], 1)
        self.assertAlmostEqual(evidence["candidate_input_correlation"], 0)
        self.assertAlmostEqual(evidence["input_residual_energy_fraction"], .0001 / 1.0001)
        decision = self.projection_artifact()[2]["streams"][1]
        self.assertEqual(decision["reason"], "independent_input_support")
        check = decision["source_support_checks"][0]
        self.assertFalse(check["direct_pass"])
        self.assertTrue(check["independent_pass"])

    @unittest.expectedFailure
    def test_unresolved_projection_artifact_must_not_publish(self):
        self.assertFalse(any(e.publication_text for e in self.partial(self.projection_artifact(), 1)))

    def test_quiet_independent_secondary_and_primary_still_publish(self):
        for amplitude in (1., .005, .001):
            with self.subTest(amplitude=amplitude):
                result = self.admit(self.primary + amplitude * self.other,
                                    (self.primary, self.other))
                self.assertFalse(result[2]["applied"])
                self.assertEqual(self.partial(result, 0)[0].publication_text, "실제 본문")
                self.assertEqual(self.partial(result, 1)[0].publication_text, "분리 후보")

    def test_repeated_unsupported_phrase_is_internal_and_discarded(self):
        assembler = SubtitleAssembler(1)
        state = SpeakerSubtitleState(1, require_final_support=True)
        ambiguous = {"candidate_input_correlation": .1, "owner_input_correlation": .98,
                     "independent_input_correlation": .28, "pair_correlation": .05}
        with patch("secondary_leakage_diagnostics.source_input_evidence", return_value=ambiguous):
            for window in range(3):
                result = self.admit(self.primary, (self.primary, self.other),
                                    texts=["실제 본문", "감사합니다!"],
                                    active=["실제 본문", state.partial_text])
                texts, vads, _ = result
                assembly = assembler.process(window, texts[1])
                event = state.process(window, assembly.utterance_hypothesis, True, 3 + 2 * window,
                                      confirmed_prefix_length=assembly.confirmed_prefix_length,
                                      source_supported=vads[1]["source_supported"], source_text=texts[1])[0]
                self.assertTrue(event.text)
                self.assertEqual(event.publication_text, "")
        self.assertEqual(state.flush(2, 7).action, "discard")

    def gap_state(self, initially_supported):
        state = SpeakerSubtitleState(0, require_final_support=True)
        state.process(0, "old", True, 3, source_supported=initially_supported, source_text="old")
        state.process(1, "old gap", True, 5, source_supported=False, source_text="gap")
        event = state.process(2, "old gap genuine", True, 7,
                              source_supported=True, source_text="genuine")[0]
        return state, event

    def test_gap_diagnostic_distinguishes_blocked_update_from_no_source_support(self):
        for initially_supported in (False, True):
            state, event = self.gap_state(initially_supported)
            self.assertEqual(event.support_update["reason"], "unsupported_prefix_gap")
            self.assertTrue(event.support_update["observed_suffix_matches"])
            self.assertEqual(event.support_update["unsupported_gap_length"],
                             3 if initially_supported else 6)
            self.assertEqual(event.publication_text, "old" if initially_supported else "")
            # Once the current source covers the gap, the state can advance.
            recovered = state.process(3, "old gap genuine", True, 9, source_supported=True,
                                      source_text="old gap genuine")[0]
            self.assertEqual(recovered.publication_text, "old gap genuine")

    @unittest.expectedFailure
    def test_unresolved_gap_must_not_hide_later_supported_speech(self):
        _, event = self.gap_state(True)
        self.assertIn("genuine", event.publication_text)

    def test_contiguous_supported_extension_is_not_frozen(self):
        state = SpeakerSubtitleState(0, require_final_support=True)
        state.process(0, "old", True, 3, source_supported=True, source_text="old")
        event = state.process(1, "old genuine", True, 5, source_supported=True, source_text="genuine")[0]
        self.assertEqual(event.publication_text, "old genuine")
        self.assertEqual(event.support_update["reason"], "current_source_covers_unretained_text")
        self.assertEqual(event.support_update["unsupported_gap_length"], 0)

    def test_source_suffix_mismatch_is_diagnosed(self):
        event = SpeakerSubtitleState(0, require_final_support=True).process(
            0, "internal hypothesis", True, 3, source_supported=True, source_text="different raw")[0]
        self.assertEqual(event.support_update["reason"], "source_not_hypothesis_suffix")
        self.assertIsNone(event.support_update["unsupported_gap_length"])
        self.assertEqual(event.publication_text, "")


if __name__ == "__main__":
    unittest.main()
