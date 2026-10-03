"""Deterministic regressions for live ghost-subtitle failure modes.

The tests supply transcripts and acoustic evidence directly.  They do not run
Whisper, ClearVoice, a GPU, or a live capture device.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from secondary_leakage_diagnostics import admit_subtitle_streams
from subtitle_assembler import SpeakerSubtitleState, SubtitleAssembler


SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 3 * SAMPLE_RATE


def vad(duration_ms: int, *, start_ms: int = 0) -> dict[str, object]:
    start = round(start_ms * SAMPLE_RATE / 1000)
    end = start + round(duration_ms * SAMPLE_RATE / 1000)
    return {
        "speech_detected": duration_ms > 0,
        "speech_duration_ms": duration_ms,
        "speech_ratio": duration_ms / 3000,
        "timestamps": [{"start": start, "end": end}] if duration_ms else [],
    }


class GhostSubtitleRegressions(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20261003)
        self.owner = rng.normal(0, 0.05, WINDOW_SAMPLES).astype(np.float32)
        self.other = rng.normal(0, 0.05, WINDOW_SAMPLES).astype(np.float32)

    def test_1_single_speaker_secondary_leakage_does_not_start_or_publish(self) -> None:
        # The first four values are from the live ghost window.  Candidate
        # residual energy and pair correlation are the conservative geometry
        # branch consistent with those supplied correlations; both fields are
        # present in each production evidence row.
        live_evidence = {
            "owner_input_correlation": 0.995933,
            "candidate_input_correlation": 0.518343,
            "independent_input_correlation": 0.857164,
            "input_residual_energy_fraction": 0.008117,
            "candidate_residual_energy_fraction": 0.659,
            "pair_correlation": 0.583,
        }
        with patch(
            "secondary_leakage_diagnostics.source_input_evidence",
            return_value=live_evidence,
        ):
            texts, effective_vads, diagnostic = admit_subtitle_streams(
                mixture=self.owner,
                speakers=(self.owner, self.other),
                transcripts=["주택 청약종합저축통장을 해지했습니다.", "가짜 두 번째 자막"],
                vad_results=[vad(3000), vad(1116)],
                active_hypotheses=["주택 청약종합저축통장을 해지했습니다.", ""],
                speaker_assignment={
                    "raw_to_logical_mapping": {"0": 0, "1": 1},
                    "active_raw_slots": [0],
                    "assignment_method": "single_active_hold",
                },
            )

        decision = diagnostic["streams"][1]
        self.assertEqual(texts[1], "")
        self.assertFalse(effective_vads[1]["speech_detected"])
        self.assertFalse(decision["source_supported"])
        self.assertTrue(decision["source_support_checks"][0]["raw_direct_pass"])
        self.assertTrue(decision["source_support_checks"][0]["raw_independent_pass"])
        self.assertTrue(decision["source_support_checks"][0]["blocked_by_owner_residual"])
        self.assertEqual(
            decision["reason"],
            "owner_explains_input_with_insufficient_residual_energy",
        )
        assembly = SubtitleAssembler(1).process(0, texts[1])
        state = SpeakerSubtitleState(1, require_final_support=True)
        self.assertEqual(
            state.process(
                0,
                assembly.utterance_hypothesis,
                effective_vads[1]["speech_detected"],
                3,
                source_supported=effective_vads[1].get("source_supported", False),
            ),
            [],
        )

    def test_live_window_003_owner_dominance_defers_secondary_publication(self) -> None:
        live_secondary_evidence = {
            "owner_input_correlation": 0.9802366582512774,
            "candidate_input_correlation": 0.8121339488672095,
            "independent_input_correlation": 0.9409841526926479,
            "input_residual_energy_fraction": 0.03913609382036889,
            "candidate_residual_energy_fraction": 0.5221252342552116,
            "pair_correlation": 0.76,
        }
        primary_evidence = {
            "owner_input_correlation": 0.8121339488672095,
            "candidate_input_correlation": 0.9802366582512774,
            "independent_input_correlation": 0.95,
            "input_residual_energy_fraction": 0.34,
            "candidate_residual_energy_fraction": 0.97,
            "pair_correlation": 0.76,
        }
        with patch(
            "secondary_leakage_diagnostics.source_input_evidence",
            side_effect=[primary_evidence, live_secondary_evidence],
        ):
            texts, effective_vads, diagnostic = admit_subtitle_streams(
                mixture=self.owner,
                speakers=(self.owner, self.other),
                transcripts=[
                    "대대가 전통적인 내지 마련수단으로 여겨졌던",
                    "아련스탄으로 여겨졌던",
                ],
                vad_results=[vad(2740), vad(1270)],
                active_hypotheses=["이전 primary 문장", ""],
                speaker_assignment={
                    "raw_to_logical_mapping": {"0": 0, "1": 1},
                    "active_raw_slots": [0],
                    "assignment_method": "overlap_continuity",
                },
            )

        decision = diagnostic["streams"][1]
        self.assertFalse(decision["suppressed"])
        self.assertTrue(decision["source_support_deferred"])
        self.assertFalse(decision["source_supported"])
        self.assertTrue(decision["speech_contained_in_owner"])
        self.assertTrue(decision["owner_candidate_conflict"])
        self.assertEqual(
            decision["reason"],
            "contained_owner_dominant_awaiting_confirmation",
        )
        self.assertEqual(texts[1], "아련스탄으로 여겨졌던")

        assembly = SubtitleAssembler(1).process(3, texts[1])
        event = SpeakerSubtitleState(1, require_final_support=True).process(
            3,
            assembly.utterance_hypothesis,
            effective_vads[1]["speech_detected"],
            9,
            confirmed_prefix_length=assembly.confirmed_prefix_length,
            source_supported=effective_vads[1]["source_supported"],
            source_text=texts[1],
        )[0]
        self.assertEqual(event.stable_text, "")
        self.assertEqual(event.tentative_text, "아련스탄으로 여겨졌던")
        self.assertEqual(event.source_supported_text, "")
        self.assertEqual(event.publication_text, "")

    def test_live_window_004_existing_secondary_leakage_is_not_promoted(self) -> None:
        secondary_evidence_003 = {
            "owner_input_correlation": 0.9802366582512774,
            "candidate_input_correlation": 0.8121339488672095,
            "independent_input_correlation": 0.9409841526926479,
            "input_residual_energy_fraction": 0.03913609382036889,
            "candidate_residual_energy_fraction": 0.5221252342552116,
            "pair_correlation": 0.76,
        }
        secondary_evidence_004 = {
            "owner_input_correlation": 0.9882876104612045,
            "candidate_input_correlation": 0.8688347640433384,
            "independent_input_correlation": 0.8963134493121728,
            "input_residual_energy_fraction": 0.023287599008882876,
            "candidate_residual_energy_fraction": 0.36763268737239524,
            "pair_correlation": 0.78,
        }
        primary_evidence = {
            "owner_input_correlation": 0.80,
            "candidate_input_correlation": 0.98,
            "independent_input_correlation": 0.95,
            "input_residual_energy_fraction": 0.36,
            "candidate_residual_energy_fraction": 0.98,
            "pair_correlation": 0.76,
        }
        assembler = SubtitleAssembler(1)
        state = SpeakerSubtitleState(1, require_final_support=True)

        with patch(
            "secondary_leakage_diagnostics.source_input_evidence",
            side_effect=[primary_evidence, secondary_evidence_003],
        ):
            texts_003, vads_003, _ = admit_subtitle_streams(
                mixture=self.owner,
                speakers=(self.owner, self.other),
                transcripts=[
                    "대대가 전통적인 내지 마련수단으로 여겨졌던",
                    "아련스탄으로 여겨졌던",
                ],
                vad_results=[vad(2740), vad(1270)],
                active_hypotheses=["이전 primary 문장", ""],
                speaker_assignment={
                    "raw_to_logical_mapping": {"0": 0, "1": 1},
                    "active_raw_slots": [0],
                    "assignment_method": "overlap_continuity",
                },
            )
        first = assembler.process(3, texts_003[1])
        first_event = state.process(
            3,
            first.utterance_hypothesis,
            vads_003[1]["speech_detected"],
            9,
            source_supported=vads_003[1]["source_supported"],
            source_text=texts_003[1],
        )[0]
        self.assertEqual(first_event.publication_text, "")

        with patch(
            "secondary_leakage_diagnostics.source_input_evidence",
            side_effect=[primary_evidence, secondary_evidence_004],
        ):
            texts_004, vads_004, diagnostic = admit_subtitle_streams(
                mixture=self.owner,
                speakers=(self.owner, self.other),
                transcripts=[
                    "수단으로 여겨졌던 청약통장을 잇따라 해제하고 있습니다.",
                    "수단으로 여겨졌던 청약통장을",
                ],
                vad_results=[vad(2900), vad(1852)],
                active_hypotheses=["primary 진행 중", state.partial_text],
                speaker_assignment={
                    "raw_to_logical_mapping": {"0": 0, "1": 1},
                    "active_raw_slots": [0],
                    "assignment_method": "overlap_continuity",
                },
            )

        decision = diagnostic["streams"][1]
        self.assertTrue(decision["existing_partial"])
        self.assertTrue(decision["speech_contained_in_owner"])
        self.assertTrue(decision["owner_candidate_conflict"])
        self.assertTrue(
            decision["cross_stream_lexical_overlap"][
                "candidate_contained_in_owner_text"
            ]
        )
        self.assertTrue(decision["suppressed"])
        self.assertFalse(decision["source_supported"])
        self.assertEqual(decision["reason"], "contained_cross_stream_leakage")
        self.assertEqual(texts_004[1], "")
        self.assertTrue(vads_004[1]["speech_detected"])
        self.assertTrue(decision["tentative_retained"])

        second = assembler.process(4, texts_004[1])
        retained = state.process(
            4,
            second.utterance_hypothesis,
            vads_004[1]["speech_detected"],
            11,
            source_supported=vads_004[1]["source_supported"],
            source_text=texts_004[1],
        )[0]
        self.assertEqual(retained.status, "partial")
        self.assertEqual(retained.source_supported_text, "")
        self.assertEqual(retained.publication_text, "")

        final = state.process(5, "", False, 13)[0]
        self.assertEqual(final.action, "discard")
        self.assertEqual(final.source_supported_text, "")
        self.assertEqual(final.publication_text, "")

    def test_2_real_two_speaker_overlap_with_independent_energy_keeps_both(self) -> None:
        mixture = self.owner + self.other
        texts, effective_vads, diagnostic = admit_subtitle_streams(
            mixture=mixture,
            speakers=(self.owner, self.other),
            transcripts=["첫 번째 화자입니다.", "두 번째 화자입니다."],
            vad_results=[vad(3000), vad(3000)],
            active_hypotheses=["", ""],
            speaker_assignment={
                "raw_to_logical_mapping": {"0": 0, "1": 1},
                "active_raw_slots": [0, 1],
                "assignment_method": "dual_source",
            },
        )

        self.assertFalse(diagnostic["applied"])
        self.assertEqual(texts, ["첫 번째 화자입니다.", "두 번째 화자입니다."])
        for speaker in (0, 1):
            self.assertTrue(effective_vads[speaker]["source_supported"])
            event = SpeakerSubtitleState(speaker, require_final_support=True).process(
                0,
                texts[speaker],
                True,
                3,
                source_supported=True,
                source_text=texts[speaker],
            )[0]
            self.assertEqual(event.publication_text, texts[speaker])

    def test_risky_contained_but_distinct_second_speaker_publishes_immediately(self) -> None:
        risky_but_supported = {
            "owner_input_correlation": 0.9802366582512774,
            "candidate_input_correlation": 0.8121339488672095,
            "independent_input_correlation": 0.9409841526926479,
            "input_residual_energy_fraction": 0.03913609382036889,
            "candidate_residual_energy_fraction": 0.5221252342552116,
            "pair_correlation": 0.30,
        }
        primary_evidence = {
            "owner_input_correlation": 0.70,
            "candidate_input_correlation": 0.98,
            "independent_input_correlation": 0.95,
            "input_residual_energy_fraction": 0.51,
            "candidate_residual_energy_fraction": 0.98,
            "pair_correlation": 0.30,
        }
        with patch(
            "secondary_leakage_diagnostics.source_input_evidence",
            side_effect=[primary_evidence, risky_but_supported],
        ):
            texts, effective_vads, diagnostic = admit_subtitle_streams(
                mixture=self.owner + 0.01 * self.other,
                speakers=(self.owner, self.other),
                transcripts=["첫 번째 화자의 본문", "서로 다른 두 번째 화자"],
                vad_results=[vad(2800), vad(1200)],
                active_hypotheses=["primary 진행 중", ""],
                speaker_assignment={
                    "raw_to_logical_mapping": {"0": 0, "1": 1},
                    "active_raw_slots": [0],
                    "assignment_method": "overlap_continuity",
                },
            )
        decision = diagnostic["streams"][1]
        self.assertEqual(decision["reason"], "independent_input_support")
        self.assertTrue(decision["source_supported"])
        self.assertFalse(
            decision["cross_stream_lexical_overlap"]["duplicate_structure"]
        )
        event = SpeakerSubtitleState(1, require_final_support=True).process(
            0,
            texts[1],
            effective_vads[1]["speech_detected"],
            3,
            source_supported=effective_vads[1]["source_supported"],
            source_text=texts[1],
        )[0]
        self.assertEqual(event.publication_text, "서로 다른 두 번째 화자")

    def test_1c_outside_owner_candidate_publishes_on_next_supported_window(self) -> None:
        evidence = {
            "owner_input_correlation": 0.60,
            "candidate_input_correlation": 0.80,
            "independent_input_correlation": 0.90,
            "input_residual_energy_fraction": 0.64,
            "candidate_residual_energy_fraction": 0.90,
            "pair_correlation": 0.20,
        }
        state = SpeakerSubtitleState(1, require_final_support=True)
        with patch(
            "secondary_leakage_diagnostics.source_input_evidence",
            return_value=evidence,
        ):
            for window in (0, 1):
                texts, effective_vads, diagnostic = admit_subtitle_streams(
                    mixture=self.owner + self.other,
                    speakers=(self.owner, self.other),
                    transcripts=["실제 발화", "확인되는 두 번째 화자"],
                    vad_results=[vad(1200), vad(1594, start_ms=1400)],
                    active_hypotheses=["실제 발화", state.partial_text],
                    speaker_assignment={
                        "raw_to_logical_mapping": {"0": 0, "1": 1},
                        "active_raw_slots": [0],
                        "assignment_method": "single_active_hold",
                    },
                )
                event = state.process(
                    window,
                    texts[1],
                    effective_vads[1]["speech_detected"],
                    3 + 2 * window,
                    source_supported=effective_vads[1]["source_supported"],
                    source_text=texts[1],
                )[0]
                if window == 0:
                    self.assertEqual(event.publication_text, "")
                    self.assertEqual(
                        diagnostic["streams"][1]["reason"],
                        "speech_outside_owner_awaiting_confirmation",
                    )
                else:
                    self.assertEqual(
                        diagnostic["streams"][1]["reason"], "existing_utterance"
                    )
                    self.assertEqual(event.publication_text, "확인되는 두 번째 화자")

    def test_1b_outside_owner_first_observation_waits_and_silence_discards_it(self) -> None:
        strong_evidence = {
            "owner_input_correlation": 0.60,
            "candidate_input_correlation": 0.80,
            "independent_input_correlation": 0.90,
            "input_residual_energy_fraction": 0.64,
            "candidate_residual_energy_fraction": 0.90,
            "pair_correlation": 0.20,
        }
        owner_vad = vad(1200)
        candidate_vad = vad(1594, start_ms=1400)
        with patch(
            "secondary_leakage_diagnostics.source_input_evidence",
            return_value=strong_evidence,
        ):
            texts, effective_vads, diagnostic = admit_subtitle_streams(
                mixture=self.owner + self.other,
                speakers=(self.owner, self.other),
                transcripts=["실제 발화", "첫 창의 미확인 후보"],
                vad_results=[owner_vad, candidate_vad],
                active_hypotheses=["실제 발화", ""],
                speaker_assignment={
                    "raw_to_logical_mapping": {"0": 0, "1": 1},
                    "active_raw_slots": [0],
                    "assignment_method": "single_active_hold",
                },
            )

        decision = diagnostic["streams"][1]
        self.assertEqual(
            decision["reason"], "speech_outside_owner_awaiting_confirmation"
        )
        self.assertTrue(decision["source_support_deferred"])
        self.assertFalse(effective_vads[1]["source_supported"])
        state = SpeakerSubtitleState(1, require_final_support=True)
        partial = state.process(
            0,
            texts[1],
            effective_vads[1]["speech_detected"],
            3,
            source_supported=effective_vads[1]["source_supported"],
            source_text=texts[1],
        )[0]
        self.assertEqual(partial.stable_text, "")
        self.assertEqual(partial.publication_text, "")
        final = state.process(1, "", False, 5)[0]
        self.assertEqual(final.action, "discard")
        self.assertEqual(final.publication_text, "")

    def test_3_existing_partial_rejects_live_476ms_unlocalized_suffix(self) -> None:
        mixture = self.owner
        candidate = self.other
        assembler = SubtitleAssembler(0)
        state = SpeakerSubtitleState(0, require_final_support=True)
        first = assembler.process(0, "검증된 정상 문장")
        state.process(
            0,
            first.utterance_hypothesis,
            True,
            3,
            confirmed_prefix_length=first.confirmed_prefix_length,
            source_supported=True,
            source_text=first.raw,
        )

        live_weak_evidence = {
            "suppressed": False,
            "available": True,
            "input_speech_rms": 0.0210937303196679,
            "input_speech_peak": 0.07,
            "input_background_rms": 0.0633979682010026,
            "weak_vad": True,
            "no_localized_burst": True,
            "rms_within_noise_bound": False,
            "peak_within_noise_bound": False,
        }
        with patch(
            "secondary_leakage_diagnostics.weak_speech_evidence",
            return_value=live_weak_evidence,
        ):
            texts, effective_vads, diagnostic = admit_subtitle_streams(
                mixture=mixture,
                speakers=(candidate, np.zeros_like(candidate)),
                transcripts=["합니다.", ""],
                vad_results=[vad(476), vad(0)],
                active_hypotheses=[state.partial_text, ""],
                speaker_assignment={},
            )
        decision = diagnostic["streams"][0]
        self.assertTrue(decision["existing_partial"])
        self.assertTrue(decision["weak_speech_check_applied"])
        self.assertFalse(decision["weak_speech_evidence"]["suppressed"])
        self.assertTrue(decision["weak_existing_tail_suppressed"])
        self.assertEqual(
            decision["reason"], "existing_partial_weak_unlocalized_tail"
        )
        self.assertEqual(texts[0], "")

        tail = assembler.process(1, texts[0])
        final = state.process(
            1,
            tail.utterance_hypothesis,
            effective_vads[0]["speech_detected"],
            5,
            confirmed_prefix_length=tail.confirmed_prefix_length,
            source_supported=effective_vads[0].get("source_supported", False),
            source_text=texts[0],
        )[0]
        self.assertEqual(final.status, "final")
        self.assertEqual(final.publication_text, "검증된 정상 문장")
        self.assertNotIn("합니다", final.text)

    def test_4_unsupported_tentative_is_discarded_on_silence(self) -> None:
        state = SpeakerSubtitleState(1, require_final_support=True)
        partial = state.process(
            0,
            "미지원 두 번째 후보",
            True,
            3,
            source_supported=False,
        )[0]
        self.assertEqual(partial.publication_text, "")
        final = state.process(1, "", False, 5)[0]
        self.assertEqual(final.action, "discard")
        self.assertEqual(final.support_update["reason"], "finalize_without_new_evidence")
        self.assertEqual(final.publication_text, "")

    def test_5_supported_subtitle_finalizes_normally_on_silence(self) -> None:
        state = SpeakerSubtitleState(0, require_final_support=True)
        partial = state.process(
            0,
            "검증된 문장",
            True,
            3,
            source_supported=True,
            source_text="검증된 문장",
        )[0]
        self.assertEqual(partial.publication_text, "검증된 문장")
        final = state.process(1, "", False, 5)[0]
        self.assertEqual(final.action, "finalize")
        self.assertEqual(final.publication_text, "검증된 문장")

    def test_6_sustained_vad_silence_creates_no_subtitle_event(self) -> None:
        states = [
            SpeakerSubtitleState(speaker, require_final_support=True)
            for speaker in (0, 1)
        ]
        for window in range(4):
            texts, effective_vads, _ = admit_subtitle_streams(
                mixture=np.zeros(WINDOW_SAMPLES, dtype=np.float32),
                speakers=(
                    np.zeros(WINDOW_SAMPLES, dtype=np.float32),
                    np.zeros(WINDOW_SAMPLES, dtype=np.float32),
                ),
                transcripts=["", ""],
                vad_results=[vad(0), vad(0)],
                active_hypotheses=[state.partial_text for state in states],
                speaker_assignment={"input_silence": True},
            )
            for speaker in (0, 1):
                self.assertEqual(
                    states[speaker].process(
                        window,
                        texts[speaker],
                        effective_vads[speaker]["speech_detected"],
                        3 + 2 * window,
                    ),
                    [],
                )

    def test_7_normal_continuous_speech_still_deduplicates_overlap(self) -> None:
        assembler = SubtitleAssembler(0)
        state = SpeakerSubtitleState(0, require_final_support=True)
        first = assembler.process(0, "오늘 주요 소식입니다")
        first_event = state.process(
            0,
            first.utterance_hypothesis,
            True,
            3,
            confirmed_prefix_length=first.confirmed_prefix_length,
            source_supported=True,
            source_text=first.raw,
        )[0]
        second = assembler.process(1, "주요 소식입니다 다음 내용입니다")
        second_event = state.process(
            1,
            second.utterance_hypothesis,
            True,
            5,
            confirmed_prefix_length=second.confirmed_prefix_length,
            source_supported=True,
            source_text=second.raw,
        )[0]

        self.assertEqual(first_event.publication_text, "오늘 주요 소식입니다")
        self.assertEqual(
            second_event.publication_text,
            "오늘 주요 소식입니다 다음 내용입니다",
        )
        self.assertEqual(second.match_type, "token_exact")


if __name__ == "__main__":
    unittest.main()
