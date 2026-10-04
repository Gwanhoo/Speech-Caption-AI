"""Signal fixtures with supplied VAD/STT, not recordings of the Windows failure.

The baseline WAVs are not in the checkout. In particular the later windows'
speech-local correlations are unknown; these tests expose admission holes,
without pretending that generated tones/noise reproduce MossFormer or Whisper.
"""
from __future__ import annotations

import unittest

import numpy as np

from secondary_leakage_diagnostics import (
    admit_subtitle_streams, build_window_diagnostic,
    secondary_transcript_suppression_reason,
)
from subtitle_assembler import SpeakerSubtitleState, SubtitleAssembler


SR = 16000
SAMPLES = 3 * SR


def vad(milliseconds):
    return {
        "speech_detected": bool(milliseconds), "speech_duration_ms": milliseconds,
        "speech_ratio": milliseconds / 3000,
        "timestamps": [{"start": 0, "end": milliseconds * 16}] if milliseconds else [],
    }


class ArtifactSourceAdmissionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(41213)
        self.primary, self.artifact, self.error = rng.normal(0, .1, (3, SAMPLES))

    def admit(self, *, text="고맙습니다", genuine=None, existing=False,
              owner=0, gain=1., vads=None, candidate=None):
        # A small owner error leaves residual input energy, independent of the
        # invented component in the other output. Shared primary speech alone
        # lets the artifact exceed the OLD direct-support correlation threshold.
        mixture = self.primary
        primary = self.primary + .0094 * self.error
        secondary = .65 * self.primary + np.sqrt(1 - .65**2) * self.artifact
        if genuine is not None:
            mixture = self.primary + genuine * self.artifact
            primary, secondary = self.primary, self.artifact
        if candidate is not None:
            secondary = candidate
        waves = [primary, secondary * gain]
        texts = ["건강관리에 주의할 것을 당부했습니다", text]
        activity = vads or [vad(2358), vad(1052)]
        active = ["정상 발화", "이전에 확인된 두 번째 발화" if existing else ""]
        if owner:
            waves.reverse(); texts.reverse(); activity = activity[::-1]; active.reverse()
        diagnostic = build_window_diagnostic(
            speaker_0=waves[0], speaker_1=waves[1], vad_results=activity, transcripts=texts,
        )
        result = admit_subtitle_streams(
            mixture=mixture, speakers=tuple(waves), transcripts=texts,
            vad_results=activity, active_hypotheses=active,
            speaker_assignment={"raw_to_logical_mapping": {"0": 0, "1": 1},
                                "active_raw_slots": [0, 1]},
            window=12,
        )
        return result, diagnostic

    def test_shared_primary_component_is_not_independent_source_support(self):
        result, diagnostic = self.admit()
        self.assertTrue(diagnostic["secondary_artifact_candidate"])
        self.assertIsNone(secondary_transcript_suppression_reason(diagnostic, "고맙습니다"))
        decision = result[2]["streams"][1]
        check = decision["source_support_checks"][0]
        self.assertGreater(check["direct_correlation"], .45)
        self.assertGreater(check["owner_input_correlation"], .99)
        self.assertLess(check["independent_correlation"], .20)
        self.assertEqual(result[0][1], "")
        self.assertFalse(result[1][1]["source_supported"])
        self.assertTrue(decision["suppressed"])
        self.assertEqual(result[0][0], "건강관리에 주의할 것을 당부했습니다")

    def test_decision_is_text_gain_polarity_and_logical_id_independent(self):
        for text in ("고맙습니다", "감사합니다", "내일 기차가 도착합니다", "arbitrary words"):
            for owner in (0, 1):
                for gain in (.001, 1., -10.):
                    with self.subTest(text=text, owner=owner, gain=gain):
                        result, _ = self.admit(text=text, owner=owner, gain=gain)
                        self.assertEqual(result[0][1 - owner], "")
                        self.assertTrue(result[0][owner])

    def test_genuine_overlap_including_very_quiet_secondary_keeps_both(self):
        for amplitude in (1., .005, .001):
            for owner in (0, 1):
                for gain in (1., -10.):
                    with self.subTest(amplitude=amplitude, owner=owner, gain=gain):
                        result, _ = self.admit(genuine=amplitude, owner=owner, gain=gain)
                        self.assertTrue(all(result[0]))
                        self.assertFalse(result[2]["applied"])
                        self.assertTrue(all(v["source_supported"] for v in result[1]))

    def test_existing_partial_cannot_bypass_current_source_contradiction(self):
        for candidate in (None, self.artifact):
            with self.subTest(shared_primary=candidate is None):
                result, _ = self.admit(existing=True, candidate=candidate)
                self.assertTrue(result[2]["streams"][1]["existing_partial"])
                self.assertEqual(result[0][1], "")
                self.assertFalse(result[1][1]["source_supported"])

    def test_repeated_windows_do_not_promote_artifact_to_partial_or_final(self):
        assembler = SubtitleAssembler(1)
        state = SpeakerSubtitleState(1, require_final_support=True)
        for window in (12, 13):
            result, _ = self.admit()
            texts, vads, _ = result
            assembly = assembler.process(window, texts[1])
            events = state.process(
                window, assembly.utterance_hypothesis, vads[1]["speech_detected"],
                3 + 2 * window, source_supported=vads[1].get("source_supported", False),
                source_text=texts[1], candidate_transition=vads[1].get("candidate_transition"),
            )
            self.assertEqual(events, [])
        self.assertIsNone(state.flush(13, 29))

    def test_invalid_or_missing_speech_evidence_does_not_trigger_new_rejection(self):
        for activity in ([vad(2358), dict(vad(1052), timestamps=[])],
                         [vad(2358), dict(vad(1052), timestamps=[{"start": -1, "end": 100}])]):
            result, _ = self.admit(vads=activity)
            self.assertFalse(result[2]["applied"])
            self.assertEqual(result[0][1], "고맙습니다")

    def test_one_genuine_interval_protects_candidate_despite_another_artifact_interval(self):
        mixture = self.primary.copy()
        primary = self.primary + .0094 * self.error
        candidate = .65 * self.primary + np.sqrt(1 - .65**2) * self.artifact
        mixture[SR:2 * SR] += .001 * self.artifact[SR:2 * SR]
        primary[SR:2 * SR] = self.primary[SR:2 * SR]
        candidate[SR:2 * SR] = self.artifact[SR:2 * SR]
        activity = [vad(3000), dict(vad(2000), timestamps=[
            {"start": 0, "end": SR}, {"start": SR, "end": 2 * SR},
        ])]
        texts, vads, diagnostic = admit_subtitle_streams(
            mixture=mixture, speakers=(primary, candidate),
            transcripts=["정상 본문입니다", "실제로 작은 목소리입니다"], vad_results=activity,
            active_hypotheses=["정상 본문", ""], speaker_assignment={},
        )
        checks = diagnostic["streams"][1]["source_support_checks"]
        self.assertTrue(checks[0]["contradicted_by_owner"])
        self.assertTrue(checks[1]["independent_pass"])
        self.assertEqual(texts[1], "실제로 작은 목소리입니다")
        self.assertTrue(vads[1]["source_supported"])
        self.assertFalse(diagnostic["applied"])

    def test_rejected_continuation_preserves_previously_supported_final(self):
        state = SpeakerSubtitleState(1, require_final_support=True)
        legitimate = "이전에 확인된 두 번째 발화"
        state.process(11, legitimate, True, 25, source_supported=True, source_text=legitimate)
        result, _ = self.admit(existing=True)
        texts, vads, _ = result
        events = state.process(
            12, texts[1], vads[1]["speech_detected"], 27,
            source_supported=vads[1]["source_supported"], source_text=texts[1],
            candidate_transition=vads[1].get("candidate_transition"),
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].status, "final")
        self.assertEqual(events[0].publication_text, legitimate)

    def test_artifact_rejection_does_not_require_owner_vad_containment(self):
        result, _ = self.admit(vads=[vad(1000), dict(vad(1052), timestamps=[
            {"start": 16000, "end": 16000 + 1052 * 16},
        ])])
        self.assertEqual(result[0][1], "")
        self.assertTrue(result[2]["streams"][1]["suppressed"])

    def test_whole_mixture_copy_is_not_evidence_against_real_primary(self):
        for gain in (1., -1.):
            mixture = self.primary + self.artifact
            texts, vads, diagnostic = admit_subtitle_streams(
                mixture=mixture, speakers=(self.primary, gain * mixture),
                transcripts=["실제 첫 번째 발화", "실제 두 번째 발화"],
                vad_results=[vad(3000), vad(3000)], active_hypotheses=["", ""],
                speaker_assignment={},
            )
            self.assertEqual(texts, ["실제 첫 번째 발화", "실제 두 번째 발화"])
            self.assertTrue(all(v["source_supported"] for v in vads))
            self.assertFalse(diagnostic["applied"])


if __name__ == "__main__":
    unittest.main()
