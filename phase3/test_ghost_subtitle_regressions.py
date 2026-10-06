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

    def lifecycle_window(self, assembler, state, window, text, evidence, *, outside=False):
        # Supplied acoustic measurements, not a claim that random audio is speech.
        primary = {
            "owner_input_correlation": .60, "candidate_input_correlation": .98,
            "independent_input_correlation": .95, "input_residual_energy_fraction": .64,
            "candidate_residual_energy_fraction": .98, "pair_correlation": .30,
        }
        with patch("secondary_leakage_diagnostics.source_input_evidence",
                   side_effect=[primary, evidence]):
            texts, vads, diagnostic = admit_subtitle_streams(
                mixture=self.owner + self.other, speakers=(self.owner, self.other),
                transcripts=["우크라이나 외무장관은 대통령실에서 회담했습니다", text],
                vad_results=[vad(1200 if outside else 2900),
                             vad(2514, start_ms=486) if outside else vad(1800)],
                active_hypotheses=["primary 진행 중", state.partial_text],
                speaker_assignment={"raw_to_logical_mapping": {"0": 0, "1": 1},
                                    "active_raw_slots": [0]},
                candidate_contexts=[{}, state.candidate_context], window=window,
            )
        if vads[1].get("candidate_transition", {}).get("restart"):
            discarded = state.finalize(window, 3 + 2 * window, "secondary_candidate_restart")
            self.assertEqual(discarded.publication_text, "")
            assembler.reset_utterance(final_text=discarded.text)
        assembly = assembler.process(window, texts[1])
        events = state.process(
            window, assembly.utterance_hypothesis, vads[1]["speech_detected"], 3 + 2 * window,
            confirmed_prefix_length=assembly.confirmed_prefix_length,
            source_supported=vads[1].get("source_supported", False), source_text=texts[1],
            candidate_transition=vads[1].get("candidate_transition"),
        )
        for event in events:
            if event.status == "final":
                assembler.reset_utterance(final_text=event.text)
        return diagnostic["streams"][1], assembly, events

    def test_latest_distinct_text_leakage_is_held_then_discarded(self):
        # Rounded measurements supplied for the latest Live window 003.
        evidence = {
            "owner_input_correlation": .9874, "candidate_input_correlation": .9144,
            "independent_input_correlation": .9002,
            "input_residual_energy_fraction": .02498,
            "candidate_residual_energy_fraction": .27718,
            "pair_correlation": .85,  # Synthetic; absent from the supplied Live excerpt.
        }
        assembler, state = SubtitleAssembler(1), SpeakerSubtitleState(1, require_final_support=True)
        decision, _, events = self.lifecycle_window(
            assembler, state, 3, "청약 통장의 예치금을 살펴보겠습니다", evidence)
        self.assertTrue(decision["owner_candidate_conflict"])
        self.assertTrue(decision["speech_contained_in_owner"])
        self.assertFalse(decision["cross_stream_lexical_overlap"]["duplicate_structure"])
        self.assertEqual(events[0].publication_text, "")
        self.assertEqual(events[0].source_supported_text, "")
        self.assertEqual(state.candidate_context["provenance"], "TENTATIVE")
        decision, _, events = self.lifecycle_window(
            assembler, state, 4, "대출 이자를 함께 계산해 보았습니다", evidence)
        self.assertTrue(decision["existing_partial"])
        self.assertEqual(events[0].action, "discard")
        self.assertEqual(events[0].publication_text, "")
        self.assertEqual(state.candidate_context["provenance"], "NONE")
        self.assertEqual(assembler.utterance_hypothesis, "")
        self.assertEqual(state.process(5, "", False, 13), [])
        self.assertIsNone(state.flush(5, 13))

    def test_latest_independent_secondary_promotes_changing_overlap_text(self):
        evidence = {
            "owner_input_correlation": .60, "candidate_input_correlation": .80,
            "independent_input_correlation": .999,
            "input_residual_energy_fraction": .64,
            "candidate_residual_energy_fraction": .90, "pair_correlation": .20,
        }  # Only independent correlation and VAD below are from the Live excerpt.
        assembler, state = SubtitleAssembler(1), SpeakerSubtitleState(1, require_final_support=True)
        decision, _, first = self.lifecycle_window(
            assembler, state, 0, "이천삼십 세대의 청약 통장 예치금은", evidence, outside=True)
        self.assertEqual(decision["reason"], "speech_outside_owner_awaiting_confirmation")
        self.assertTrue(decision["source_support_deferred"])
        self.assertEqual(first[0].publication_text, "")
        decision, assembly, second = self.lifecycle_window(
            assembler, state, 1, "청약 통장 예치금은 십만 원입니다", evidence, outside=True)
        self.assertTrue(decision["source_supported"])
        self.assertNotEqual(assembly.utterance_hypothesis, assembly.raw)
        self.assertTrue(second[0].publication_text)
        self.assertEqual(second[0].publication_text, assembly.utterance_hypothesis)
        self.assertEqual(state.candidate_context["provenance"], "VALIDATED")
        final = state.process(2, "", False, 7)[0]
        self.assertEqual(final.action, "finalize")
        self.assertEqual(final.publication_text, second[0].publication_text)
        self.assertEqual(state.candidate_context["provenance"], "NONE")

    def test_candidate_after_window_gap_restarts_without_publishing_old_text(self):
        evidence = {
            "owner_input_correlation": .60, "candidate_input_correlation": .80,
            "independent_input_correlation": .99, "input_residual_energy_fraction": .64,
            "candidate_residual_energy_fraction": .90, "pair_correlation": .20,
        }
        assembler, state = SubtitleAssembler(1), SpeakerSubtitleState(1, require_final_support=True)
        self.lifecycle_window(assembler, state, 0, "이전 창에서 보류한 문장", evidence, outside=True)
        decision, _, held = self.lifecycle_window(
            assembler, state, 2, "새 발화의 대출 이자는", evidence, outside=True)
        self.assertTrue(decision["candidate_restart"])
        self.assertEqual(held[0].publication_text, "")
        self.assertEqual(held[0].utterance_id, 2)
        _, _, confirmed = self.lifecycle_window(
            assembler, state, 3, "대출 이자는 낮아졌습니다", evidence, outside=True)
        self.assertTrue(confirmed[0].publication_text)
        self.assertNotIn("이전 창", confirmed[0].publication_text)
        self.assertIn("새 발화", confirmed[0].publication_text)
        self.assertEqual(state.flush(3, 9).publication_text, confirmed[0].publication_text)

    def test_candidate_with_unbacked_prefix_requires_fresh_confirmation(self):
        independent = {
            "owner_input_correlation": .60, "candidate_input_correlation": .80,
            "independent_input_correlation": .99, "input_residual_energy_fraction": .64,
            "candidate_residual_energy_fraction": .90, "pair_correlation": .20,
        }
        ambiguous = dict(independent, candidate_input_correlation=.10, independent_input_correlation=.28)
        assembler, state = SubtitleAssembler(1), SpeakerSubtitleState(1, require_final_support=True)
        self.lifecycle_window(assembler, state, 0, "미지원 첫 관측", ambiguous, outside=True)
        decision, _, held = self.lifecycle_window(
            assembler, state, 1, "청약 통장 예치금은", independent, outside=True)
        self.assertTrue(decision["candidate_restart"])
        self.assertEqual(held[0].publication_text, "")
        _, _, confirmed = self.lifecycle_window(
            assembler, state, 2, "청약 통장 예치금은 십만 원입니다", independent, outside=True)
        self.assertEqual(confirmed[0].publication_text, "청약 통장 예치금은 십만 원입니다")
        self.assertNotIn("미지원", assembler.assembled)

    def test_short_independent_reply_outside_owner_publishes_and_finalizes(self):
        state = SpeakerSubtitleState(1, require_final_support=True)
        texts, vads, diagnostic = admit_subtitle_streams(
            mixture=self.owner + self.other, speakers=(self.owner, self.other),
            transcripts=["계속되는 주요 소식", "네"],
            vad_results=[vad(1200), vad(320, start_ms=2000)],
            active_hypotheses=["primary 진행 중", ""], speaker_assignment={},
            candidate_contexts=[{}, state.candidate_context], window=0,
        )
        self.assertFalse(diagnostic["streams"][1]["speech_contained_in_owner"])
        self.assertTrue(vads[1]["source_supported"])
        partial = state.process(
            0, texts[1], True, 3, source_supported=vads[1]["source_supported"],
            source_text=texts[1], candidate_transition=vads[1].get("candidate_transition"),
        )[0]
        self.assertEqual(partial.publication_text, "네")
        self.assertEqual(state.process(1, "", False, 5)[0].publication_text, "네")

    def test_candidate_confirmation_does_not_authorize_later_unsupported_suffix(self):
        evidence = {
            "owner_input_correlation": .60, "candidate_input_correlation": .80,
            "independent_input_correlation": .99, "input_residual_energy_fraction": .64,
            "candidate_residual_energy_fraction": .90, "pair_correlation": .20,
        }
        assembler, state = SubtitleAssembler(1), SpeakerSubtitleState(1, require_final_support=True)
        self.lifecycle_window(assembler, state, 0, "청약 통장 예치금은", evidence, outside=True)
        _, _, events = self.lifecycle_window(
            assembler, state, 1, "청약 통장 예치금은 십만 원입니다", evidence, outside=True)
        supported = events[0].publication_text
        state.process(2, supported + " 미지원 구간", True, 7,
                      source_supported=False, source_text="미지원 구간")
        later = state.process(3, supported + " 미지원 구간 새 발화", True, 9,
                              source_supported=True, source_text="새 발화")[0]
        self.assertEqual(later.support_update["reason"], "unsupported_prefix_gap")
        self.assertEqual(later.publication_text, supported)
        self.assertEqual(state.process(4, "", False, 11)[0].publication_text, supported)

    def test_tentative_confirmation_cannot_launder_copied_unsupported_prefix(self):
        state = SpeakerSubtitleState(1, require_final_support=True)
        state.process(0, "미지원 prefix", True, 3, source_supported=False)
        state.process(1, "미지원 prefix 실제 관측", True, 5, source_supported=False,
                      source_text="실제 관측",
                      candidate_transition={"action": "hold", "independent_support": True})
        self.assertFalse(state.candidate_context["pending_text_supported"])
        candidate = state.process(2, "미지원 prefix 실제 관측 새 내용", True, 7,
                                  source_supported=True, source_text="새 내용",
                                  candidate_transition={"action": "validate", "independent_support": True})[0]
        self.assertEqual(candidate.publication_text, "")
        self.assertEqual(candidate.support_update["reason"], "unsupported_prefix_gap")
        self.assertEqual(state.flush(2, 7).action, "discard")

    def test_explicit_confirmation_handles_revised_seam_without_global_suffix_bypass(self):
        # A synthetic assembled seam, not an assertion about which SUPPORT
        # reason occurred in the unavailable two-news Live log.
        for deferred in (False, True):
            with self.subTest(explicit_deferred_observation=deferred):
                state = SpeakerSubtitleState(1, require_final_support=True)
                state.process(
                    0, "청약 통장", True, 3, source_supported=False, source_text="청약 통장",
                    candidate_transition={"action": "hold", "independent_support": True}
                    if deferred else None,
                )
                event = state.process(
                    1, "청약 통장 예치금은 십만 원", True, 5,
                    source_supported=True, source_text="통장의 예치금은 십만 원",
                    candidate_transition={"action": "validate", "independent_support": True},
                )[0]
                if deferred:
                    self.assertEqual(event.publication_text, "청약 통장 예치금은 십만 원")
                    self.assertEqual(event.support_update["reason"], "secondary_candidate_confirmed")
                    self.assertEqual(event.support_update["coverage_reason_before_confirmation"],
                                     "source_not_hypothesis_suffix")
                else:
                    self.assertEqual(event.publication_text, "")
                    self.assertEqual(event.support_update["reason"], "source_not_hypothesis_suffix")

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

    def test_risky_contained_distinct_text_also_requires_confirmation(self) -> None:
        # Preserve the d2efaad evidence: distinct STT is no longer proof of
        # independence when the very same owner-conflict geometry is present.
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
        self.assertEqual(decision["reason"], "contained_owner_dominant_awaiting_confirmation")
        self.assertTrue(decision["source_support_deferred"])
        self.assertFalse(decision["source_supported"])
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
        self.assertEqual(event.publication_text, "")
        self.assertEqual(event.source_supported_text, "")

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

    def test_3a_live_silence_540ms_artifact_never_publishes_or_finalizes(self) -> None:
        rng = np.random.default_rng(20261006)
        mixture = rng.uniform(-0.000565, 0.000565, WINDOW_SAMPLES).astype(np.float32)
        start, end = 34336, 42976
        mixture[start:end] = rng.uniform(-0.000753, 0.000753, end - start)
        candidate = rng.normal(0, 1, WINDOW_SAMPLES).astype(np.float32)
        correlation_only = {
            "owner_input_correlation": 0.4366949,
            "candidate_input_correlation": 0.7228728,
            "independent_input_correlation": 0.7711960,
            "input_residual_energy_fraction": 0.80,
            "candidate_residual_energy_fraction": 0.90,
            "pair_correlation": 0.10,
        }
        candidate_vad = {
            "speech_detected": True,
            "speech_duration_ms": 540,
            "speech_ratio": 0.18,
            "timestamps": [{"start": start, "end": end}],
        }
        with patch(
            "secondary_leakage_diagnostics.source_input_evidence",
            return_value=correlation_only,
        ):
            texts, effective_vads, diagnostic = admit_subtitle_streams(
                mixture=mixture,
                speakers=(np.zeros_like(mixture), candidate),
                transcripts=["", "무음에서 생성된 임의의 문장"],
                vad_results=[vad(0), candidate_vad],
                active_hypotheses=["", ""],
                speaker_assignment={
                    "raw_to_logical_mapping": {"0": 0, "1": 1},
                    "active_raw_slots": [1],
                    "assignment_method": "bootstrap",
                },
            )

        decision = diagnostic["streams"][1]
        weak = decision["weak_speech_evidence"]
        self.assertFalse(weak["weak_vad"])
        self.assertTrue(weak["no_localized_burst"])
        self.assertTrue(weak["rms_within_noise_bound"])
        self.assertTrue(weak["peak_within_noise_bound"])
        self.assertTrue(decision["source_support_checks"][0]["raw_direct_pass"])
        self.assertTrue(decision["source_support_checks"][0]["raw_independent_pass"])
        self.assertTrue(decision["suppressed"])
        self.assertFalse(decision["source_supported"])
        self.assertEqual(texts[1], "")
        self.assertFalse(effective_vads[1]["speech_detected"])

        assembler = SubtitleAssembler(1)
        state = SpeakerSubtitleState(1, require_final_support=True)
        assembly = assembler.process(3, texts[1])
        self.assertEqual(
            state.process(
                3,
                assembly.utterance_hypothesis,
                effective_vads[1]["speech_detected"],
                9,
                source_supported=effective_vads[1]["source_supported"],
                source_text=texts[1],
            ),
            [],
        )
        self.assertEqual(state.process(6, "", False, 15), [])
        self.assertIsNone(state.flush(6, 15))

    def test_3b_low_volume_localized_speech_is_not_suppressed(self) -> None:
        rng = np.random.default_rng(20261007)
        mixture = rng.uniform(-0.0002, 0.0002, WINDOW_SAMPLES).astype(np.float32)
        start, end = 34336, 42976
        phase = np.arange(end - start)
        mixture[start:end] = (0.0009 * np.sin(2 * np.pi * phase / 80)).astype(np.float32)
        evidence = {
            "owner_input_correlation": 0.20,
            "candidate_input_correlation": 0.72,
            "independent_input_correlation": 0.77,
            "input_residual_energy_fraction": 0.80,
            "candidate_residual_energy_fraction": 0.90,
            "pair_correlation": 0.10,
        }
        candidate_vad = {
            "speech_detected": True,
            "speech_duration_ms": 540,
            "speech_ratio": 0.18,
            "timestamps": [{"start": start, "end": end}],
        }
        with patch(
            "secondary_leakage_diagnostics.source_input_evidence", return_value=evidence
        ):
            texts, effective_vads, diagnostic = admit_subtitle_streams(
                mixture=mixture,
                speakers=(np.zeros_like(mixture), mixture.copy()),
                transcripts=["", "작지만 국소화된 실제 발화"],
                vad_results=[vad(0), candidate_vad],
                active_hypotheses=["", ""],
                speaker_assignment={},
            )

        decision = diagnostic["streams"][1]
        self.assertTrue(decision["weak_speech_evidence"]["rms_within_noise_bound"])
        self.assertTrue(decision["weak_speech_evidence"]["peak_within_noise_bound"])
        self.assertFalse(decision["weak_speech_evidence"]["no_localized_burst"])
        self.assertFalse(decision["suppressed"])
        self.assertTrue(effective_vads[1]["source_supported"])
        self.assertEqual(texts[1], "작지만 국소화된 실제 발화")

    def test_3c_residual_vad_protects_low_volume_independent_speech(self) -> None:
        rng = np.random.default_rng(20261008)
        mixture = rng.uniform(-0.000565, 0.000565, WINDOW_SAMPLES).astype(np.float32)
        start, end = 34336, 42976
        mixture[start:end] = rng.uniform(-0.000753, 0.000753, end - start)
        candidate_vad = {
            "speech_detected": True,
            "speech_duration_ms": 540,
            "speech_ratio": 0.18,
            "timestamps": [{"start": start, "end": end}],
            "input_residual_speech": {
                "version": 1,
                "available": True,
                "sample_count": WINDOW_SAMPLES,
                "reason": "measured",
                "speech_detected": True,
                "timestamps": [{"start": start, "end": end}],
            },
        }
        evidence = {
            "owner_input_correlation": 0.98,
            "candidate_input_correlation": 0.72,
            "independent_input_correlation": 0.77,
            "input_residual_energy_fraction": 0.20,
            "candidate_residual_energy_fraction": 0.90,
            "pair_correlation": 0.10,
        }
        with patch(
            "secondary_leakage_diagnostics.source_input_evidence", return_value=evidence
        ):
            texts, effective_vads, diagnostic = admit_subtitle_streams(
                mixture=mixture,
                speakers=(mixture.copy(), rng.normal(size=WINDOW_SAMPLES).astype(np.float32)),
                transcripts=["", "작지만 residual에서 확인된 실제 발화"],
                vad_results=[vad(0), candidate_vad],
                active_hypotheses=["", ""],
                speaker_assignment={},
            )

        decision = diagnostic["streams"][1]
        self.assertTrue(decision["weak_speech_evidence"]["no_localized_burst"])
        self.assertTrue(decision["weak_speech_evidence"]["independent_speech_evidence"])
        self.assertFalse(decision["suppressed"])
        self.assertTrue(effective_vads[1]["source_supported"])
        self.assertTrue(texts[1])

    def test_3d_strong_316ms_transient_is_held_then_discarded_on_silence(self) -> None:
        samples = np.arange(WINDOW_SAMPLES)
        mixture = (0.06776 * np.sqrt(2) * np.sin(2 * np.pi * samples / 97)).astype(
            np.float32
        )
        start, end = 16000, 16000 + 5056
        burst = 0.15226 * np.sqrt(2) * np.sin(
            2 * np.pi * np.arange(end - start) / 41
        )
        burst[0] = 0.52701
        mixture[start:end] = burst
        candidate_vad = {
            "speech_detected": True,
            "speech_duration_ms": 316,
            "speech_ratio": 316 / 3000,
            "timestamps": [{"start": start, "end": end}],
            "input_residual_speech": {
                "version": 1,
                "available": True,
                "sample_count": WINDOW_SAMPLES,
                "reason": "measured",
                "speech_detected": False,
                "timestamps": [],
            },
        }
        source_only = {
            "owner_input_correlation": 0.30,
            "candidate_input_correlation": 0.94443,
            "independent_input_correlation": 0.95578,
            "input_residual_energy_fraction": 0.80,
            "candidate_residual_energy_fraction": 0.90,
            "pair_correlation": 0.10,
        }
        state = SpeakerSubtitleState(1, require_final_support=True)
        with patch(
            "secondary_leakage_diagnostics.source_input_evidence",
            return_value=source_only,
        ):
            texts, effective_vads, diagnostic = admit_subtitle_streams(
                mixture=mixture,
                speakers=(np.zeros_like(mixture), mixture.copy()),
                transcripts=["", "실제 발화가 아닌 임의의 추론 문장"],
                vad_results=[vad(0), candidate_vad],
                active_hypotheses=["", ""],
                speaker_assignment={},
                candidate_contexts=[{}, state.candidate_context],
                window=43,
            )

        decision = diagnostic["streams"][1]
        weak = decision["weak_speech_evidence"]
        self.assertTrue(weak["weak_vad"])
        self.assertFalse(weak["no_localized_burst"])
        self.assertFalse(weak["rms_within_noise_bound"])
        self.assertFalse(weak["peak_within_noise_bound"])
        self.assertFalse(weak["independent_speech_evidence"])
        self.assertTrue(decision["source_support_checks"][0]["raw_direct_pass"])
        self.assertTrue(decision["source_support_checks"][0]["raw_independent_pass"])
        self.assertEqual(decision["candidate_transition"], "hold")
        self.assertFalse(decision["source_supported"])

        partial = state.process(
            43,
            texts[1],
            effective_vads[1]["speech_detected"],
            89,
            source_supported=effective_vads[1]["source_supported"],
            source_text=texts[1],
            candidate_transition=effective_vads[1]["candidate_transition"],
        )[0]
        self.assertEqual(partial.candidate_provenance, "TENTATIVE")
        self.assertEqual(partial.publication_text, "")
        final = state.process(44, "", False, 91)[0]
        self.assertEqual(final.action, "discard")
        self.assertEqual(final.publication_text, "")

    def test_3e_strong_316ms_speech_with_residual_vad_publishes_normally(self) -> None:
        samples = np.arange(WINDOW_SAMPLES)
        mixture = (0.06776 * np.sqrt(2) * np.sin(2 * np.pi * samples / 97)).astype(
            np.float32
        )
        start, end = 16000, 16000 + 5056
        mixture[start:end] = (
            0.15226
            * np.sqrt(2)
            * np.sin(2 * np.pi * np.arange(end - start) / 41)
        )
        candidate_vad = {
            "speech_detected": True,
            "speech_duration_ms": 316,
            "speech_ratio": 316 / 3000,
            "timestamps": [{"start": start, "end": end}],
            "input_residual_speech": {
                "version": 1,
                "available": True,
                "sample_count": WINDOW_SAMPLES,
                "reason": "measured",
                "speech_detected": True,
                "timestamps": [{"start": start, "end": end}],
            },
        }
        source_and_speech = {
            "owner_input_correlation": 0.30,
            "candidate_input_correlation": 0.94443,
            "independent_input_correlation": 0.95578,
            "input_residual_energy_fraction": 0.80,
            "candidate_residual_energy_fraction": 0.90,
            "pair_correlation": 0.10,
        }
        with patch(
            "secondary_leakage_diagnostics.source_input_evidence",
            return_value=source_and_speech,
        ):
            texts, effective_vads, diagnostic = admit_subtitle_streams(
                mixture=mixture,
                speakers=(np.zeros_like(mixture), mixture.copy()),
                transcripts=["", "네"],
                vad_results=[vad(0), candidate_vad],
                active_hypotheses=["", ""],
                speaker_assignment={},
                window=43,
            )

        decision = diagnostic["streams"][1]
        self.assertTrue(decision["weak_speech_evidence"]["weak_vad"])
        self.assertTrue(decision["weak_speech_evidence"]["independent_speech_evidence"])
        self.assertEqual(decision["candidate_transition"], "keep")
        self.assertTrue(effective_vads[1]["source_supported"])
        state = SpeakerSubtitleState(1, require_final_support=True)
        partial = state.process(
            43,
            texts[1],
            True,
            89,
            source_supported=True,
            source_text=texts[1],
        )[0]
        self.assertEqual(partial.publication_text, "네")
        self.assertEqual(state.process(44, "", False, 91)[0].publication_text, "네")

    def test_3f_real_speech_restarts_after_deferred_transient_without_ghost_prefix(self) -> None:
        source = {
            "owner_input_correlation": 0.30,
            "candidate_input_correlation": 0.94443,
            "independent_input_correlation": 0.95578,
            "input_residual_energy_fraction": 0.80,
            "candidate_residual_energy_fraction": 0.90,
            "pair_correlation": 0.10,
        }
        transient = {
            "suppressed": False,
            "available": True,
            "weak_vad": True,
            "no_localized_burst": False,
            "rms_within_noise_bound": False,
            "peak_within_noise_bound": False,
            "independent_speech_evidence": False,
        }
        speech = dict(transient, weak_vad=False, independent_speech_evidence=True)
        assembler = SubtitleAssembler(1)
        state = SpeakerSubtitleState(1, require_final_support=True)

        with patch(
            "secondary_leakage_diagnostics.source_input_evidence", return_value=source
        ), patch(
            "secondary_leakage_diagnostics.weak_speech_evidence",
            return_value=transient,
        ):
            first_texts, first_vads, _ = admit_subtitle_streams(
                mixture=self.owner,
                speakers=(np.zeros_like(self.owner), self.other),
                transcripts=["", "미확인 효과음 후보"],
                vad_results=[vad(0), vad(316)],
                active_hypotheses=["", ""],
                speaker_assignment={},
                candidate_contexts=[{}, state.candidate_context],
                window=43,
            )
        first = assembler.process(43, first_texts[1])
        held = state.process(
            43,
            first.utterance_hypothesis,
            first_vads[1]["speech_detected"],
            89,
            source_supported=first_vads[1]["source_supported"],
            source_text=first_texts[1],
            candidate_transition=first_vads[1]["candidate_transition"],
        )[0]
        self.assertEqual(held.publication_text, "")

        with patch(
            "secondary_leakage_diagnostics.source_input_evidence", return_value=source
        ), patch(
            "secondary_leakage_diagnostics.weak_speech_evidence",
            return_value=speech,
        ):
            texts, effective_vads, diagnostic = admit_subtitle_streams(
                mixture=self.owner,
                speakers=(np.zeros_like(self.owner), self.other),
                transcripts=["", "이제 실제로 말한 정상 문장"],
                vad_results=[vad(0), vad(1200)],
                active_hypotheses=["", state.partial_text],
                speaker_assignment={},
                candidate_contexts=[{}, state.candidate_context],
                window=44,
            )

        decision = diagnostic["streams"][1]
        self.assertTrue(decision["candidate_restart"])
        self.assertEqual(decision["candidate_transition"], "keep")
        discarded = state.finalize(44, 91, "secondary_candidate_restart")
        self.assertEqual(discarded.action, "discard")
        self.assertEqual(discarded.publication_text, "")
        assembler.reset_utterance(final_text=discarded.text)
        actual = assembler.process(44, texts[1])
        published = state.process(
            44,
            actual.utterance_hypothesis,
            effective_vads[1]["speech_detected"],
            91,
            source_supported=effective_vads[1]["source_supported"],
            source_text=texts[1],
            candidate_transition=effective_vads[1]["candidate_transition"],
        )[0]
        self.assertEqual(published.publication_text, "이제 실제로 말한 정상 문장")
        self.assertNotIn("효과음", published.publication_text)

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
