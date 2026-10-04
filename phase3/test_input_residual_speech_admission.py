"""Actual repository A.wav separation replay plus explicit signal counterexamples.

The stored GPU result is NOT the unavailable Windows single_speaker w12.
Unit tests replay measured Silero evidence; the GPU validator recomputes it.
"""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np

from secondary_leakage_diagnostics import (
    admit_subtitle_streams, attach_input_residual_speech_evidence,
)
from subtitle_assembler import SpeakerSubtitleState, SubtitleAssembler


FIXTURE = Path(__file__).parent / "test_audio/source_admission/single_a_mossformer_window0"


def load_fixture():
    metadata = json.loads(FIXTURE.with_suffix(".json").read_text())
    with np.load(FIXTURE.with_suffix(".npz"), allow_pickle=False) as data:
        waves = [data[key].copy() for key in ("mixture", "owner", "candidate")]
    return waves, metadata


def activity(start=0, end=48000):
    return dict(speech_detected=True, speech_duration_ms=(end-start)/16,
                speech_ratio=(end-start)/48000, timestamps=[dict(start=start, end=end)])


class InputResidualSpeechAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.waves, self.meta = load_fixture()

    def admit(self, *, text=None, reverse=False, gain=1., context=None, vads=None):
        x, p, c = self.waves
        signals = [p, c * gain]
        texts = [self.meta["raw_transcripts"][0], text or self.meta["raw_transcripts"][1]]
        vads = copy.deepcopy(vads if vads is not None else self.meta["vad_results"])
        active = [texts[0], texts[1] if context else ""]
        contexts = [{"provenance": "VALIDATED"}, context or {}]
        if reverse:
            signals.reverse(); texts.reverse(); vads.reverse(); active.reverse(); contexts.reverse()
        return admit_subtitle_streams(
            mixture=x, speakers=tuple(signals), transcripts=texts, vad_results=vads,
            active_hypotheses=active, candidate_contexts=contexts, window=12,
            speaker_assignment={"raw_to_logical_mapping": {"0": int(reverse), "1": int(not reverse)},
                                "active_raw_slots": [0]},
        )

    def test_actual_mossformer_residual_only_false_support_is_rejected(self):
        self.assertEqual(hashlib.sha256(FIXTURE.with_suffix(".npz").read_bytes()).hexdigest(),
                         self.meta["fixture_sha256"])
        legacy_vads = copy.deepcopy(self.meta["vad_results"])
        for vad in legacy_vads:
            vad.pop("input_residual_speech", None)
        legacy = self.admit(vads=legacy_vads)[2]["streams"][1]
        self.assertEqual(legacy["reason"], "independent_input_support")
        self.assertTrue(legacy["candidate_independent_support"])
        self.assertFalse(legacy["suppressed"])
        for check in legacy["source_support_checks"]:
            self.assertFalse(check["raw_direct_pass"])
            self.assertTrue(check["raw_independent_pass"])
            self.assertFalse(check["contradicted_by_owner"])
            self.assertFalse(check["blocked_by_owner_residual"])
        texts, vads, evidence = self.admit()
        decision = evidence["streams"][1]
        self.assertEqual(decision["reason"], "owner_residual_contains_no_candidate_speech")
        self.assertFalse(decision["candidate_independent_support"])
        self.assertFalse(vads[1]["source_supported"])
        self.assertEqual(texts, [self.meta["raw_transcripts"][0], ""])

    def test_actual_artifact_is_content_gain_polarity_and_slot_invariant(self):
        for text in ("감사합니다", "내일 기차가 도착합니다", "An arbitrary English sentence"):
            for reverse in (False, True):
                for gain in (.001, 1., -10.):
                    with self.subTest(text=text, reverse=reverse, gain=gain):
                        texts, _, evidence = self.admit(text=text, reverse=reverse, gain=gain)
                        self.assertEqual(texts[1-int(reverse)], "")
                        self.assertEqual(texts[int(reverse)], self.meta["raw_transcripts"][0])
                        self.assertTrue(evidence["streams"][1-int(reverse)]["suppressed"])

    def test_partial_and_repetition_never_replace_current_source_evidence(self):
        for provenance in ("NONE", "TENTATIVE", "VALIDATED"):
            context = dict(provenance=provenance, window=11, independent_support=True,
                           pending_text_supported=True)
            texts, vads, evidence = self.admit(context=context)
            self.assertEqual(texts[1], "")
            self.assertEqual(evidence["streams"][1]["candidate_transition"], "discard")
        assembler = SubtitleAssembler(1)
        state = SpeakerSubtitleState(1, require_final_support=True)
        for window in range(12, 16):
            texts, vads, _ = self.admit()
            event = assembler.process(window, texts[1])
            self.assertEqual(event.raw, "")
            self.assertEqual(event.new, "")
            self.assertEqual(state.process(window, event.utterance_hypothesis, False, 2*window+3,
                source_supported=vads[1]["source_supported"], source_text=texts[1],
                candidate_transition=vads[1]["candidate_transition"]), [])
        self.assertEqual(event.session_text, "")
        self.assertIsNone(state.flush(16, 35))

    def test_invalid_residual_measurements_fail_open(self):
        for invalid in (None, {}, {"available": False},
                        dict(self.meta["vad_results"][1]["input_residual_speech"], sample_count=1),
                        dict(self.meta["vad_results"][1]["input_residual_speech"], timestamps=[{"start": -1, "end": 10}]),
                        dict(self.meta["vad_results"][1]["input_residual_speech"], speech_detected=True)):
            with self.subTest(invalid=invalid):
                vads = copy.deepcopy(self.meta["vad_results"])
                vads[1]["input_residual_speech"] = invalid
                self.assertTrue(self.admit(vads=vads)[0][1])

    def test_one_supported_interval_protects_whole_candidate(self):
        vads = copy.deepcopy(self.meta["vad_results"])
        interval = vads[1]["timestamps"][1]
        vads[1]["input_residual_speech"].update(speech_detected=True, timestamps=[interval])
        texts, _, evidence = self.admit(vads=vads)
        self.assertTrue(texts[1])
        checks = evidence["streams"][1]["source_support_checks"]
        self.assertTrue(checks[0]["residual_nonspeech"])
        self.assertTrue(checks[1]["independent_pass"])

    def test_genuine_weak_sources_use_normalized_input_residual_not_output_volume(self):
        rng = np.random.default_rng(507)
        p, q = rng.normal(0, .1, (2, 48000))
        for amplitude in (1., .005, .001, .00001):
            for gain in (.001, 1., -10.):
                with self.subTest(amplitude=amplitude, gain=gain):
                    x = gain * (p + amplitude*q)
                    signals = (gain*p, -7*q)
                    seen = []
                    def detect(residual):
                        seen.append(residual)
                        self.assertAlmostEqual(float(np.sqrt(np.mean(residual**2))), .1, places=6)
                        self.assertGreater(abs(np.corrcoef(q, residual)[0, 1]), .999)
                        return activity()
                    vads = attach_input_residual_speech_evidence(x, signals, [activity(), activity()], detect)
                    texts, _, evidence = admit_subtitle_streams(
                        mixture=x, speakers=signals, transcripts=["첫 화자", "작은 실제 발화"],
                        vad_results=vads, active_hypotheses=["첫 화자", ""], speaker_assignment={})
                    self.assertEqual(texts, ["첫 화자", "작은 실제 발화"])
                    self.assertFalse(evidence["applied"])
                    if amplitude < 1:
                        self.assertEqual(len(seen), 1)

    def test_detector_failure_and_degenerate_projection_are_unavailable(self):
        x, p, c = self.waves
        detector = Mock(side_effect=RuntimeError("unavailable"))
        vads = attach_input_residual_speech_evidence(x, (p,c), self.meta["vad_results"], detector)
        self.assertEqual(vads[1]["input_residual_speech"]["reason"], "residual_vad_error")
        self.assertTrue(self.admit(vads=vads)[0][1])
        detector.reset_mock()
        vads = attach_input_residual_speech_evidence(x, (x,c), self.meta["vad_results"], detector)
        self.assertEqual(vads[1]["input_residual_speech"]["reason"], "degenerate_input_residual")
        detector.assert_not_called()

    def test_real_source_leaked_into_owner_is_not_rejected_as_cancellation(self):
        rng = np.random.default_rng(109)
        p, q = rng.normal(0, .1, (2, 48000))
        mixture, owner = p + .001*q, p + .01*q
        def detect(residual):
            # The owner overestimates q; the residual has inverted polarity,
            # but still contains the real second source, not non-speech error.
            self.assertLess(np.corrcoef(q, residual)[0,1], -.999)
            return activity()
        vads = attach_input_residual_speech_evidence(
            mixture, (owner,q), [activity(), activity()], detect)
        texts, _, evidence = admit_subtitle_streams(
            mixture=mixture, speakers=(owner,q), transcripts=["첫 화자", "작은 두 번째 화자"],
            vad_results=vads, active_hypotheses=["첫 화자", ""], speaker_assignment={})
        self.assertEqual(texts, ["첫 화자", "작은 두 번째 화자"])
        self.assertFalse(evidence["applied"])


if __name__ == "__main__":
    unittest.main()
