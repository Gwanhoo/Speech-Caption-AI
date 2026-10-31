"""3s/2s synthetic audio + supplied STT; never claims recognition/GPU accuracy."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from secondary_leakage_diagnostics import (
    build_window_diagnostic, resolve_subtitle_fragments, subtitle_overlap_evidence,
)
from speaker_tracking import PersistentSpeakerTracker
from subtitle_assembler import SpeakerSubtitleState, SubtitleAssembler

SR = 16000
SAMPLES = 3 * SR


def vad(*intervals):
    duration = sum(end - start for start, end in intervals)
    return {
        "speech_detected": bool(intervals), "speech_ratio": duration / 3,
        "speech_duration_ms": duration * 1000,
        "timestamps": [{"start": int(start * SR), "end": int(end * SR)} for start, end in intervals],
    }


class CaptionContinuityTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(429)

    def audio(self):
        return self.rng.normal(0, .1, SAMPLES).astype(np.float32)

    def resolve(self, primary=None, secondary=None, texts=None, vads=None,
                previous=None, active=None):
        primary = self.audio() if primary is None else primary
        secondary = -primary if secondary is None else secondary
        texts = texts or ["랜덤 챔피언 혀당인데요", "룰렛 돌려서"]
        vads = vads or [vad((0, 2)), vad((2, 3))]
        diagnostic = build_window_diagnostic(
            speaker_0=primary, speaker_1=secondary, vad_results=vads, transcripts=texts,
        )
        return resolve_subtitle_fragments(
            speakers=(primary, secondary), transcripts=texts, vad_results=vads,
            diagnostic=diagnostic,
            previous_texts=previous or ["지금 오신 분들 위해서 랜덤 챔피언", ""],
            active_hypotheses=active or ["지금 오신 분들 위해서 랜덤 챔피언", ""],
        )

    def test_a_c_d_routed_fragment_swap_correction_and_silence_final(self):
        tracker = PersistentSpeakerTracker(overlap_samples=SR)
        assemblers = [SubtitleAssembler(i) for i in (0, 1)]
        states = [SpeakerSubtitleState(i) for i in (0, 1)]
        primary = self.audio()
        last_audio, last_vads = None, None
        all_events = []
        for window in range(4):
            if window:
                primary = np.concatenate((primary[-SR:], self.audio()[:2 * SR]))
            if window == 0:
                raw = (primary, self.audio())
                texts = ["어 지금 오신 분들 위해서 랜덤 챔피언", ""]
                vads = [vad((0, 3)), vad()]
            elif window == 1:
                residual = primary.copy()
                residual[:2 * SR] *= .2
                # Mimic per-output RMS normalization: level cannot classify it.
                residual *= np.sqrt(np.mean(primary ** 2) / np.mean(residual ** 2))
                raw = (primary, residual)
                texts = ["랜덤 챔피언 혀당인데요", "룰렛 돌려서"]
                vads = [vad((0, 2)), vad((2, 3))]
            elif window == 2:
                raw = (self.audio(), primary)  # a real raw permutation
                texts = ["", "제가 돌려서 AB 챔피언 3개 AP 챔피언 3개"]
                vads = [vad(), vad((0, 3))]
            else:
                primary = np.zeros(SAMPLES, dtype=np.float32)
                raw, texts, vads = (primary, primary), ["", ""], [vad(), vad()]
            assignment = tracker.assign(window=window, raw_speakers=raw, mixture=primary)
            mapping = assignment.diagnostic["raw_to_logical_mapping"]
            logical_texts, logical_vads = [None, None], [None, None]
            for i in (0, 1):
                logical_texts[mapping[str(i)]] = texts[i]
                logical_vads[mapping[str(i)]] = vads[i]
            diagnostic = build_window_diagnostic(
                speaker_0=assignment.speakers[0], speaker_1=assignment.speakers[1],
                vad_results=logical_vads, transcripts=logical_texts,
            )
            resolved, effective_vads, routing = resolve_subtitle_fragments(
                speakers=assignment.speakers, transcripts=logical_texts,
                vad_results=logical_vads, diagnostic=diagnostic,
                previous_texts=[a.previous for a in assemblers],
                active_hypotheses=[s.partial_text for s in states],
            )
            if window == 1:
                self.assertTrue(routing["applied"])
                self.assertEqual(routing["reason"], "ordered_continuation_on_same_waveform")
                self.assertEqual(assignment.diagnostic["excluded_duplicate_tail_reference"], 1)
            if window == 2:
                self.assertEqual(mapping, {"0": 1, "1": 0})
            for speaker in (0, 1):
                evidence = subtitle_overlap_evidence(
                    last_audio[speaker], assignment.speakers[speaker],
                    last_vads[speaker], effective_vads[speaker], SR,
                ) if last_audio is not None else {}
                assembly = assemblers[speaker].process(
                    window, resolved[speaker],
                    shared_speech=evidence.get("shared_speech"),
                    supported_tail_revision=evidence.get("supported_tail_revision", False),
                )
                if window == 2 and speaker == 0:
                    self.assertEqual(assembly.match_type, "fuzzy_replace")
                    self.assertEqual(assembly.new, "AB 챔피언 3개 AP 챔피언 3개")
                    self.assertFalse(assembly.duplicate_only)
                events = states[speaker].process(
                    window, assembly.utterance_hypothesis,
                    effective_vads[speaker]["speech_detected"], 3 + 2 * window,
                    confirmed_prefix_length=assembly.confirmed_prefix_length,
                )
                all_events.extend(events)
                if events and events[-1].status == "final":
                    assemblers[speaker].reset_utterance()
            last_audio, last_vads = assignment.speakers, effective_vads
        self.assertEqual({event.speaker for event in all_events}, {0})
        final = all_events[-1]
        self.assertEqual(final.finalize_reason, "vad_silence")
        self.assertEqual(final.text, "어 지금 오신 분들 위해서 랜덤 챔피언 혀당인데요 제가 돌려서 AB 챔피언 3개 AP 챔피언 3개")
        self.assertEqual(all_events[-2].action, "replace")
        self.assertEqual(states[0].hypothesis_history, [])
        self.assertEqual(assemblers[0].utterance_hypothesis, "")

    def test_b_c_two_independent_speakers_keep_both_streams_through_swap(self):
        tracker = PersistentSpeakerTracker(overlap_samples=SR)
        assemblers = [SubtitleAssembler(i) for i in (0, 1)]
        states = [SpeakerSubtitleState(i) for i in (0, 1)]
        a, b = self.audio(), self.audio()
        for window in range(3):
            if window:
                a = np.concatenate((a[-SR:], self.audio()[:2 * SR]))
                b = np.concatenate((b[-SR:], self.audio()[:2 * SR]))
            swap = window % 2 == 1
            assignment = tracker.assign(window=window, raw_speakers=(b, a) if swap else (a, b), mixture=a + b)
            self.assertEqual(assignment.diagnostic["raw_to_logical_mapping"], {"0": int(swap), "1": int(not swap)})
            self.assertIsNone(assignment.diagnostic["excluded_duplicate_tail_reference"])
            texts = ["랜덤 챔피언 이야기입니다", "저는 다른 게임을 합니다"]
            vads = [vad((0, 3)), vad((0, 3))]
            diagnostic = build_window_diagnostic(speaker_0=assignment.speakers[0], speaker_1=assignment.speakers[1], vad_results=vads, transcripts=texts)
            resolved, effective, evidence = resolve_subtitle_fragments(
                speakers=assignment.speakers, transcripts=texts, vad_results=vads, diagnostic=diagnostic,
                previous_texts=[a.previous for a in assemblers], active_hypotheses=[s.partial_text for s in states],
            )
            self.assertFalse(evidence["applied"])
            for i in (0, 1):
                assembly = assemblers[i].process(window, resolved[i])
                events = states[i].process(window, assembly.utterance_hypothesis, effective[i]["speech_detected"], 3 + 2 * window)
                self.assertEqual(events[0].text, texts[i])

    def test_b_independent_secondary_even_quiet_or_same_words_is_retained(self):
        for text in ("룰렛 돌려서", "챔피언 혀당인데요", "네", "진짜 진짜"):
            with self.subTest(text=text):
                texts = ["랜덤 챔피언 혀당인데요", text]
                result, _, evidence = self.resolve(secondary=self.audio() * .001, texts=texts)
                self.assertEqual(result, texts)
                self.assertFalse(evidence["applied"])

    def test_b_established_secondary_and_overlapping_different_text_retained(self):
        for overrides in (
            {"active": ["랜덤 챔피언", "두 번째 화자"], "previous": ["랜덤 챔피언", "두 번째 화자"]},
            {"vads": [vad((0, 3)), vad((1, 2))]},
        ):
            with self.subTest(overrides=overrides):
                result, _, evidence = self.resolve(**overrides)
                self.assertEqual(result[1], "룰렛 돌려서")
                self.assertFalse(evidence["applied"])

    def test_no_timing_no_continuity_or_long_fragment_is_not_routed(self):
        for overrides in (
            {"vads": [dict(vad((0, 2)), timestamps=[]), vad((2, 3))]},
            {"previous": ["관계없는 문장입니다", ""]},
            {"texts": ["랜덤 챔피언 혀당인데요", "완전히 다른 사람이 길게 하는 이야기입니다"]},
        ):
            with self.subTest(overrides=overrides):
                _, _, evidence = self.resolve(**overrides)
                self.assertFalse(evidence["applied"])

    def test_duplicate_fragment_and_symmetric_owner_do_not_mutate_raw(self):
        texts = ["랜덤 챔피언 혀당인데요", "챔피언 혀당인데요"]
        vads = [vad((0, 3)), vad((2, 3))]
        original_vads = copy.deepcopy(vads)
        resolved, effective, evidence = self.resolve(texts=texts, vads=vads)
        self.assertTrue(evidence["applied"])
        self.assertEqual(resolved, [texts[0], ""])
        self.assertFalse(effective[1]["speech_detected"])
        self.assertEqual(vads, original_vads)
        self.assertEqual(texts[1], "챔피언 혀당인데요")
        primary = self.audio()
        texts = ["룰렛 돌려서", "랜덤 챔피언 혀당인데요"]
        result, _, evidence = self.resolve(
            primary=-primary, secondary=primary, texts=texts,
            vads=[vad((2, 3)), vad((0, 2))],
            previous=["", "랜덤 챔피언"], active=["", "랜덤 챔피언"],
        )
        self.assertEqual(result, ["", "랜덤 챔피언 혀당인데요 룰렛 돌려서"])
        self.assertEqual(evidence["owner"], 1)

    def test_e_intentional_repetition_outside_shared_speech_is_preserved(self):
        primary = self.audio()
        following = np.concatenate((primary[-SR:], self.audio()[:2 * SR]))
        evidence = subtitle_overlap_evidence(primary, following, vad((0, 1)), vad((1, 3)), SR)
        self.assertFalse(evidence["shared_speech"])
        for text in ("가 가", "진짜 진짜", "123 123", "123456", "룰렛 돌려서"):
            with self.subTest(text=text):
                assembler = SubtitleAssembler(0)
                assembler.process(0, text)
                event = assembler.process(1, text, shared_speech=evidence["shared_speech"])
                self.assertEqual(event.utterance_hypothesis, text + " " + text)
                self.assertFalse(event.duplicate_only)
        event = SubtitleAssembler(0).process(0, "가 가 진짜 진짜 3개 3개 123 123")
        self.assertEqual(event.utterance_hypothesis, event.raw)

    def test_window_10_to_11_continuous_overlap_supports_bounded_revision(self):
        # The VAD coordinates are from gui_last_run windows 10/11. The saved
        # report has no waveform arrays, so only the shared waveform is synthetic.
        previous = self.audio()
        current = np.concatenate((previous[-SR:], self.audio()[:2 * SR]))
        previous_vad = {"timestamps": [{"start": 31264, "end": 48000}]}
        current_vad = {"timestamps": [{"start": 32, "end": 48000}]}

        evidence = subtitle_overlap_evidence(
            previous, current, previous_vad, current_vad, SR,
        )

        self.assertTrue(evidence["shared_speech"])
        self.assertFalse(evidence["previous_tail_in_overlap"])
        self.assertTrue(evidence["continuous_tail_geometry_supported"])
        self.assertEqual(
            evidence["tail_revision_geometry"],
            "continuous_tail_across_overlap",
        )
        self.assertEqual(evidence["shared_tail_interval"], [32, SR])
        self.assertGreater(evidence["continuous_tail_coverage"], 0.99)
        self.assertTrue(evidence["supported_tail_revision"])
        assembler = SubtitleAssembler(0)
        assembler.process(10, "네, 현재까지 큰 교육을")
        event = assembler.process(
            11,
            "현재까지 큰 교통 혼잡은 발생하지 않았습니다.",
            shared_speech=evidence["shared_speech"],
            supported_tail_revision=evidence["supported_tail_revision"],
        )
        self.assertEqual(event.match_type, "supported_tail_replace")
        self.assertEqual(
            event.utterance_hypothesis,
            "네, 현재까지 큰 교통 혼잡은 발생하지 않았습니다.",
        )
        self.assertEqual(event.utterance_hypothesis.count("현재까지"), 1)

    def test_overlap_revision_gate_requires_timing_and_waveform_evidence(self):
        previous = self.audio()
        shared_current = np.concatenate((previous[-SR:], self.audio()[:2 * SR]))
        cases = (
            ("new_start_in_overlap", previous, shared_current, vad((2.4, 3)), vad((.4, 3)), True),
            ("no_shared_speech", previous, shared_current, vad((0, 1)), vad((1, 3)), False),
            ("low_correlation", previous, self.audio(), vad((0, 3)), vad((0, 3)), False),
            ("insufficient_continuous_coverage", previous, shared_current, vad((0, 3)), vad((.2, 3)), False),
            ("boundary_touch_only", previous, shared_current, vad((0, 2)), vad((0, 3)), False),
        )
        for name, old_audio, new_audio, old_vad, new_vad, expected in cases:
            with self.subTest(name=name):
                evidence = subtitle_overlap_evidence(
                    old_audio, new_audio, old_vad, new_vad, SR,
                )
                self.assertEqual(evidence["supported_tail_revision"], expected)

    def test_shared_waveform_and_vad_do_not_prove_predicate_identity(self):
        # Synthetic samples establish geometry only, not speech recognition.
        from test_general_subtitle_lifecycle import GeneralSubtitleHarness

        previous = self.audio()
        current = np.concatenate((previous[-SR:], self.audio()[:2 * SR]))
        old = "오늘 중앙 도서관에서는 휴관했습니다."
        raw = "중앙 도서관에서는 다음 주부터 정상 운영을 시작합니다."
        for audio, old_vad, new_vad, expected in (
            (current, vad((0, 3)), vad((0, 3)), True),
            (self.audio(), vad((0, 3)), vad((0, 3)), False),
            (current, vad((0, 1)), vad((1, 3)), False),
            (current, vad((0, 3)), vad((.2, 3)), False),
        ):
            with self.subTest(expected=expected, new_vad=new_vad):
                evidence = subtitle_overlap_evidence(previous, audio, old_vad, new_vad, SR)
                self.assertEqual(evidence["supported_tail_revision"], expected)
                harness = GeneralSubtitleHarness()
                harness.speech(0, old)
                assembly, events = harness.speech(
                    1, raw, shared_speech=evidence["shared_speech"],
                    supported_tail_revision=evidence["supported_tail_revision"],
                )
                self.assertEqual(assembly.rollover_text, old)
                self.assertEqual([e.publication_text for e in events], [old, raw])
                harness.silence(2)
                self.assertEqual(harness.state.final_segments, [old, raw])

    def test_supported_overlap_preserves_a_genuine_new_repetition(self):
        previous = self.audio()
        current = np.concatenate((previous[-SR:], self.audio()[:2 * SR]))
        evidence = subtitle_overlap_evidence(
            previous, current, vad((0, 3)), vad((0, 3)), SR,
        )
        self.assertTrue(evidence["supported_tail_revision"])
        text = "다시 한번 시작하겠습니다"
        assembler = SubtitleAssembler(0)
        assembler.process(0, text)

        event = assembler.process(
            1,
            text + " " + text,
            shared_speech=evidence["shared_speech"],
            supported_tail_revision=evidence["supported_tail_revision"],
        )

        self.assertEqual(event.match_type, "token_exact")
        self.assertEqual(event.utterance_hypothesis.count(text), 2)

    def test_d_revision_requires_timing_adjacency_and_new_content(self):
        for window, support, following in (
            (1, False, "제가 돌려서 다음 이야기"),
            (2, True, "제가 돌려서 다음 이야기"),
            (1, True, "제가 돌려서"),
            (1, True, "제가 123456 다음 이야기"),
        ):
            with self.subTest(window=window, support=support, following=following):
                assembler = SubtitleAssembler(0)
                assembler.process(0, "룰렛 돌려서")
                event = assembler.process(window, following, supported_tail_revision=support)
                self.assertEqual(event.utterance_hypothesis, "룰렛 돌려서 " + following)

    def test_timing_evidence_rejects_unrelated_audio_and_invalid_intervals(self):
        primary = self.audio()
        evidence = subtitle_overlap_evidence(primary, self.audio(), vad((2, 3)), vad((0, 3)), SR)
        self.assertFalse(evidence["supported_tail_revision"])
        following = np.concatenate((primary[-SR:], self.audio()[:2 * SR]))
        wrong_segment = subtitle_overlap_evidence(primary, following, vad((0, 2.2), (2.9, 3)), vad((0, .2)), SR)
        self.assertTrue(wrong_segment["shared_speech"])
        self.assertFalse(wrong_segment["supported_tail_revision"])
        for stamps in ([{"start": -1, "end": SR}], [{"start": 0, "end": SAMPLES + 1}], ["bad"]):
            evidence = subtitle_overlap_evidence(primary, primary, dict(vad((2, 3)), timestamps=stamps), vad((0, 3)), SR)
            self.assertIsNone(evidence["shared_speech"])


if __name__ == "__main__":
    unittest.main()
