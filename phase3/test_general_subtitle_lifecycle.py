from __future__ import annotations

import unittest

from subtitle_assembler import (
    SpeakerSubtitleState,
    SubtitleAssembler,
    normalize_for_matching,
    reconcile_adjacent_sentence_head_revision,
)


def source_fragment(event) -> str:
    if (
        event.match_type
        in {"fuzzy_replace", "supported_tail_replace", "supported_sentence_revision"}
        or not event.new_fragment
    ):
        return event.raw_fragment
    return event.new_fragment


class GeneralSubtitleHarness:
    """Small production-order harness: assembler -> state -> publication fields."""

    def __init__(self) -> None:
        self.assembler = SubtitleAssembler(0, enable_sentence_lifecycle=True)
        self.state = SpeakerSubtitleState(0, require_final_support=True)

    def speech(
        self,
        window: int,
        raw: str,
        *,
        source_supported: bool = True,
        shared_speech: bool | None = None,
        supported_tail_revision: bool = False,
    ):
        assembly = self.assembler.process(
            window,
            raw,
            shared_speech=shared_speech,
            supported_tail_revision=supported_tail_revision,
        )
        events = []
        if assembly.rollover_text and self.state.partial_text:
            final = self.state.finalize(window, 3 + 2 * window, assembly.rollover_reason or "")
            self.assembler.commit_finalized_text(final.text)
            events.append(final)
        events.extend(
            self.state.process(
                window,
                assembly.utterance_hypothesis,
                True,
                3 + 2 * window,
                confirmed_prefix_length=assembly.confirmed_prefix_length,
                source_supported=source_supported,
                source_text=source_fragment(assembly),
            )
        )
        return assembly, events

    def silence(self, window: int):
        assembly = self.assembler.process(window, "", shared_speech=False)
        events = self.state.process(window, "", False, 3 + 2 * window)
        if events and events[-1].status == "final":
            self.assembler.reset_utterance(final_text=events[-1].text)
        return assembly, events


class SupportedPredicateRevisionTests(unittest.TestCase):
    """Supplied RAW text plus explicit synthetic evidence, not an audio replay."""

    old = "오늘 서울 도심에서는 날씨였습니다."
    raw = "서울 도심에서는 아침부터 많은 시민들이 출근길에"
    expected = "오늘 서울 도심에서는 아침부터 많은 시민들이 출근길에"

    def test_h1_same_topic_must_preserve_both_sentences(self):
        for old, raw in (
            ("오늘 중앙 도서관에서는 휴관했습니다.",
             "중앙 도서관에서는 다음 주부터 정상 운영을 시작합니다."),
            ("어제 지역 박물관에서는 폐관했습니다.",
             "지역 박물관에서는 다음 달부터 새로운 전시를 시작합니다."),
        ):
            with self.subTest(old=old):
                harness = GeneralSubtitleHarness()
                harness.speech(0, old)
                assembly, events = harness.speech(
                    1, raw, shared_speech=True, supported_tail_revision=True,
                )
                self.assertEqual(assembly.rollover_text, old)
                self.assertEqual([e.publication_text for e in events], [old, raw])
                harness.silence(2)
                self.assertEqual(harness.state.final_segments, [old, raw])

    def test_m1_preserves_confirmed_prefix_but_remains_unresolved(self):
        harness = GeneralSubtitleHarness()
        prefix = "오늘 서울 도심에서는"
        harness.speech(2, prefix)
        first, _ = harness.speech(3, self.old, shared_speech=True)
        self.assertEqual(first.confirmed_prefix_length, 9)
        assembly, events = harness.speech(
            4, self.raw, shared_speech=True, supported_tail_revision=True,
        )
        # This asserts the conservative contract, NOT successful STT repair.
        self.assertEqual(assembly.rollover_text, self.old)
        self.assertEqual([e.status for e in events], ["final", "partial"])
        self.assertEqual([e.publication_text for e in events], [self.old, self.raw])
        self.assertTrue(events[0].publication_text.startswith(prefix))

    def test_reported_early_final_remains_unresolved_without_identity_evidence(self):
        harness = GeneralSubtitleHarness()
        harness.speech(3, self.old)
        assembly, events = harness.speech(
            4, self.raw, shared_speech=True, supported_tail_revision=True,
        )
        self.assertEqual(assembly.utterance_hypothesis, self.raw)
        self.assertEqual(assembly.rollover_text, self.old)
        self.assertEqual([e.status for e in events], ["final", "partial"])
        self.assertEqual([e.utterance_id for e in events], [1, 2])
        self.assertEqual([e.publication_text for e in events], [self.old, self.raw])
        _, events = harness.speech(5, "출근길에 나섰습니다.", shared_speech=True)
        expected = self.raw + " 나섰습니다."
        self.assertEqual(events[-1].publication_text, expected)
        _, events = harness.silence(6)
        self.assertEqual(events[0].publication_text, expected)
        self.assertEqual(harness.state.final_segments, [self.old, expected])

    def test_ambiguous_predicates_preserve_both_observations(self):
        for old, raw in (
            ("지금 중앙 도서관에서는 닫혔습니다.",
             "중앙 도서관에서는 학생들이 함께 자료를 찾고 있습니다."),
            ("먼저 안내드립니다. 오늘 연구 모임에서는 끝났습니다.",
             "연구 모임에서는 새로운 실험 방법을 논의하고 있습니다."),
        ):
            with self.subTest(old=old):
                harness = GeneralSubtitleHarness()
                harness.speech(0, old)
                assembly, events = harness.speech(
                    1, raw, shared_speech=True, supported_tail_revision=True,
                )
                self.assertEqual(assembly.utterance_hypothesis, raw)
                self.assertEqual([e.status for e in events], ["final", "partial"])
                self.assertEqual([e.publication_text for e in events], [old, raw])
                harness.silence(2)
                self.assertEqual(harness.state.final_segments, [old, raw])

    def test_missing_acoustic_support_or_window_gap_keeps_sentence_boundary(self):
        for window, shared, supported in (
            (4, True, False), (4, False, True), (4, None, True), (5, True, True),
        ):
            with self.subTest(window=window, shared=shared, supported=supported):
                harness = GeneralSubtitleHarness()
                harness.speech(3, self.old)
                _, events = harness.speech(
                    window, self.raw, shared_speech=shared,
                    supported_tail_revision=supported,
                )
                self.assertEqual([e.status for e in events], ["final", "partial"])
                self.assertEqual(events[0].publication_text, self.old)
                self.assertEqual(events[1].publication_text, self.raw)

    def test_confirmed_prefix_is_never_rewritten(self):
        assembler = SubtitleAssembler(0, enable_sentence_lifecycle=True)
        assembler.process(3, self.old)
        assembler.confirmed_prefix_length = len(normalize_for_matching(self.old))
        event = assembler.process(4, self.raw, shared_speech=True, supported_tail_revision=True)
        self.assertEqual(event.rollover_text, self.old)

    def test_ambiguous_anchors_and_distinct_sentences_keep_normal_final(self):
        for old, raw in (
            ("오늘 예산안을 검토했다.", "예산안을 다시 설명합니다."),
            ("오늘 서울도심에서는 날씨였습니다.", "서울도심에서는 아침부터 많은 시민들이 출근길에"),
            (self.old, "서울 도심에서는 출발"),
            ("오늘 서울 도심에서는 비가 왔습니다.", self.raw),
            ("오늘 서울 도심에서는 마쳤습니다. 안내입니다.", self.raw),
            ("오늘 서울. 도심에서는 날씨였습니다.", self.raw),
            (self.old, "서울. 도심에서는 아침부터 많은 시민들이 출근길에"),
            ("오늘 서울 도심에서는 서울 도심에서는 날씨였습니다.", self.raw),
            (self.old, "서울 도심에서는 아침부터 서울 도심에서는 사람들이 이동합니다."),
            ("오늘 날씨는 맑았습니다.", "다음 소식을 전해 드립니다."),
        ):
            with self.subTest(old=old, raw=raw):
                harness = GeneralSubtitleHarness()
                harness.speech(0, old)
                assembly, events = harness.speech(
                    1, raw, shared_speech=True, supported_tail_revision=True,
                )
                self.assertEqual(assembly.rollover_text, old)
                self.assertEqual([e.status for e in events], ["final", "partial"])
                self.assertEqual(events[0].publication_text, old)
                self.assertEqual(events[1].publication_text, raw)

    def test_repeated_utterance_after_silence_and_short_speech_survive(self):
        harness = GeneralSubtitleHarness()
        for window, text in ((0, self.old), (2, self.old), (4, "네.")):
            _, events = harness.speech(window, text)
            self.assertEqual(events[0].publication_text, text)
            _, events = harness.silence(window + 1)
            self.assertEqual(events[0].publication_text, text)
        self.assertEqual(harness.state.final_segments, [self.old, self.old, "네."])

    def test_revision_cannot_publish_unsupported_source(self):
        harness = GeneralSubtitleHarness()
        harness.speech(3, self.old, source_supported=False)
        _, events = harness.speech(
            4, self.raw, shared_speech=True, supported_tail_revision=True,
            source_supported=False,
        )
        self.assertEqual(events[-1].publication_text, "")
        _, events = harness.silence(5)
        self.assertEqual(events[0].publication_text, "")
        self.assertEqual(harness.state.final_segments, [])

    def test_supported_revision_does_not_launder_unobserved_prefix(self):
        harness = GeneralSubtitleHarness()
        harness.speech(3, self.old, source_supported=False)
        # Test State coverage for a hypothetical revision independently of the
        # withdrawn whole-predicate assembler rule. Copied "오늘" has no support.
        events = harness.state.process(
            4, self.expected, True, 11, confirmed_prefix_length=0,
            source_supported=True, source_text=self.raw,
        )
        self.assertEqual(events[-1].publication_text, "")
        self.assertEqual(events[-1].support_update["reason"], "unsupported_prefix_gap")

    def test_revision_does_not_rewrite_finalized_history(self):
        harness = GeneralSubtitleHarness()
        harness.speech(0, self.old)
        harness.silence(1)
        assembly, events = harness.speech(
            2, self.raw, shared_speech=True, supported_tail_revision=True,
        )
        self.assertEqual(harness.state.final_segments, [self.old])
        self.assertEqual(assembly.utterance_hypothesis, self.raw)
        self.assertEqual(events[-1].utterance_id, 2)
        self.assertEqual(events[-1].publication_text, self.raw)

    def test_supplied_clipped_head_and_fuzzy_particle_stay_intact(self):
        for old, raw, expected in (
            ("시민들은 대중교통을 이용하거나",
             "공교통을 이용하거나 도로 상황을 확인하며 이동하고 있습니다.",
             "시민들은 대중교통을 이용하거나 도로 상황을 확인하며 이동하고 있습니다."),
            ("이어졌지만 오후부터는 일부 지역의 구름",
             "일부 지역에 구름이 많아질 것으로 예상됩니다.",
             "이어졌지만 오후부터는 일부 지역에 구름이 많아질 것으로 예상됩니다."),
            # D's exact W012/W013 RAW was not supplied; this is synthetic.
            ("네, 현재까지 큰 교통 혼잡은 발생하지",
             "큰 교통 혼잡은 발생하지 않았습니다.",
             "네, 현재까지 큰 교통 혼잡은 발생하지 않았습니다."),
        ):
            with self.subTest(old=old):
                harness = GeneralSubtitleHarness()
                harness.speech(0, old)
                _, events = harness.speech(1, raw, shared_speech=True, supported_tail_revision=True)
                self.assertEqual(events[-1].publication_text, expected)
                _, events = harness.silence(2)
                self.assertEqual(events[0].publication_text, expected)


class GeneralSubtitleLifecycleTests(unittest.TestCase):
    def test_1_normal_sliding_overlap_stays_one_natural_utterance(self) -> None:
        harness = GeneralSubtitleHarness()
        harness.speech(0, "오늘 서울 도심에서는 아침부터")
        assembly, events = harness.speech(
            1,
            "아침부터 많은 시민들이 출근길에 나섰습니다.",
            shared_speech=True,
        )

        expected = "오늘 서울 도심에서는 아침부터 많은 시민들이 출근길에 나섰습니다."
        self.assertEqual(assembly.utterance_hypothesis, expected)
        self.assertEqual(events[-1].publication_text, expected)
        self.assertEqual(assembly.rollover_text, "")
        self.assertEqual(expected.count("아침부터"), 1)

    def test_2_premature_ending_is_revised_instead_of_finalized(self) -> None:
        harness = GeneralSubtitleHarness()
        harness.speech(0, "오전에는 비교적 맑은 날씨가 이어졌습니다.")
        assembly, events = harness.speech(
            1,
            "날씨가 이어졌지만 오후부터는 일부 지역에",
            shared_speech=True,
            supported_tail_revision=True,
        )

        expected = "오전에는 비교적 맑은 날씨가 이어졌지만 오후부터는 일부 지역에"
        self.assertEqual(assembly.match_type, "supported_sentence_revision")
        self.assertEqual(assembly.rollover_text, "")
        self.assertEqual(assembly.utterance_hypothesis, expected)
        self.assertEqual(events[-1].status, "partial")
        self.assertEqual(events[-1].utterance_id, 1)
        self.assertEqual(events[-1].publication_text, expected)

    def test_3_confirmed_ending_finalizes_before_new_utterance(self) -> None:
        harness = GeneralSubtitleHarness()
        first = "오늘 준비한 소식은 여기까지입니다."
        second = "내일은 새로운 소식을 전해드리겠습니다."
        harness.speech(0, first)
        assembly, events = harness.speech(
            1,
            f"{first} {second}",
            shared_speech=True,
        )

        self.assertEqual([event.status for event in events], ["final", "partial"])
        self.assertEqual(events[0].text, first)
        self.assertEqual(events[0].finalize_reason, "sentence_boundary_confirmed")
        self.assertEqual(events[1].utterance_id, 2)
        self.assertEqual(events[1].publication_text, second)
        self.assertEqual(assembly.utterance_hypothesis, second)
        self.assertNotIn(first, events[1].publication_text)

    def test_4_unsupported_ghost_is_still_discarded(self) -> None:
        harness = GeneralSubtitleHarness()
        _, partials = harness.speech(0, "감사합니다.", source_supported=False)
        _, rollover = harness.speech(
            1,
            "실제로 관측된 새 소식입니다.",
            source_supported=True,
            shared_speech=False,
        )

        self.assertEqual(partials[0].publication_text, "")
        self.assertEqual(rollover[0].action, "discard")
        self.assertEqual(rollover[0].publication_text, "")
        self.assertEqual(rollover[1].publication_text, "실제로 관측된 새 소식입니다.")
        self.assertEqual(harness.state.final_segments, [])

    def test_5_source_supported_subtitle_is_published_and_finalized(self) -> None:
        harness = GeneralSubtitleHarness()
        text = "시청해 주셔서 감사합니다."
        _, partials = harness.speech(0, text, source_supported=True)
        _, finals = harness.silence(1)

        self.assertEqual(partials[0].publication_text, text)
        self.assertEqual(finals[0].action, "finalize")
        self.assertEqual(finals[0].publication_text, text)

    def test_6_current_hypothesis_is_bounded_to_latest_utterance(self) -> None:
        harness = GeneralSubtitleHarness()
        sentences = [
            "첫 번째 독립 문장입니다.",
            "두 번째 독립 문장입니다.",
            "세 번째 독립 문장입니다.",
            "네 번째 독립 문장입니다.",
        ]
        finals = []
        for window, sentence in enumerate(sentences):
            assembly, events = harness.speech(
                window,
                sentence,
                shared_speech=False if window else None,
            )
            finals.extend(event.text for event in events if event.status == "final")
            self.assertEqual(assembly.utterance_hypothesis, sentence)
            self.assertEqual(harness.state.partial_text, sentence)
            for old in sentences[:window]:
                self.assertNotIn(old, assembly.utterance_hypothesis)

        _, events = harness.silence(len(sentences))
        finals.extend(event.text for event in events if event.status == "final")
        self.assertEqual(finals, sentences)

    def test_7_window_6_to_7_revises_ending_before_final(self) -> None:
        harness = GeneralSubtitleHarness()
        harness.speech(6, "오후부터는 일부 지역에 구름이 많아졌습니다.")

        assembly, events = harness.speech(
            7,
            "구름이 많아질 것으로 예상됩니다.",
            shared_speech=True,
            supported_tail_revision=False,
        )

        corrected = "오후부터는 일부 지역에 구름이 많아질 것으로 예상됩니다."
        self.assertEqual(assembly.match_type, "sentence_head_revision")
        self.assertEqual(assembly.rollover_text, "")
        self.assertEqual(assembly.utterance_hypothesis, corrected)
        self.assertEqual([event.status for event in events], ["partial"])
        self.assertEqual(events[0].utterance_id, 1)
        self.assertEqual(events[0].publication_text, corrected)

        _, following = harness.speech(
            8,
            "시민들은 대중교통을 이용하고 있습니다.",
            shared_speech=True,
        )
        self.assertEqual([event.status for event in following], ["final", "partial"])
        self.assertEqual(following[0].text, corrected)
        self.assertNotIn("많아졌습니다", following[0].text)
        self.assertNotIn(corrected, following[1].publication_text)

    def test_8_window_13_to_14_repairs_clipped_head_before_final(self) -> None:
        harness = GeneralSubtitleHarness()
        harness.speech(12, "한편 전문가들은 갑작스러운")
        harness.speech(
            13,
            "인가들은 갑작스러운 기운 변화에 대비해 건강을 받았다.",
            shared_speech=True,
        )

        assembly, events = harness.speech(
            14,
            "비해 건강관리에 주의할 것을 당부했습니다.",
            shared_speech=True,
            supported_tail_revision=False,
        )

        corrected = (
            "한편 전문가들은 갑작스러운 기운 변화에 대비해 "
            "건강관리에 주의할 것을 당부했습니다."
        )
        self.assertEqual(assembly.match_type, "sentence_head_revision")
        self.assertEqual(assembly.rollover_text, "")
        self.assertEqual(assembly.utterance_hypothesis, corrected)
        self.assertEqual([event.status for event in events], ["partial"])
        self.assertEqual(events[0].utterance_id, 1)
        self.assertEqual(events[0].publication_text, corrected)

        _, following = harness.speech(
            15,
            "당부했습니다. 그럼요. 외출하기 전",
            shared_speech=True,
        )
        self.assertEqual([event.status for event in following], ["final", "partial"])
        self.assertEqual(following[0].text, corrected)
        self.assertEqual(following[1].publication_text, "그럼요. 외출하기 전")
        self.assertNotIn("건강을 받았다", following[0].text)

    def test_9_related_new_sentence_is_not_mistaken_for_revision(self) -> None:
        harness = GeneralSubtitleHarness()
        first = "오늘 예산안을 검토했다."
        second = "예산안을 다시 설명합니다."
        harness.speech(0, first)

        assembly, events = harness.speech(
            1,
            second,
            shared_speech=True,
            supported_tail_revision=False,
        )

        self.assertNotEqual(assembly.match_type, "sentence_head_revision")
        self.assertEqual([event.status for event in events], ["final", "partial"])
        self.assertEqual(events[0].text, first)
        self.assertEqual(events[1].publication_text, second)

    def test_sentence_head_revision_preserves_independent_leading_speech(self) -> None:
        previous = (
            "첫 번째 소식은 이미 전했습니다. "
            "오후부터는 일부 지역에 구름이 많아졌습니다."
        )
        current = "구름이 많아질 것으로 예상됩니다."
        expected = (
            "첫 번째 소식은 이미 전했습니다. "
            "오후부터는 일부 지역에 구름이 많아질 것으로 예상됩니다."
        )

        revision = reconcile_adjacent_sentence_head_revision(
            previous, previous, current, 6, 0
        )
        self.assertIsNotNone(revision)
        self.assertEqual(revision[0], expected)

        assembler = SubtitleAssembler(0, enable_sentence_lifecycle=True)
        assembler.process(0, previous)
        event = assembler.process(1, current, shared_speech=True)
        self.assertEqual(event.match_type, "sentence_head_revision")
        self.assertEqual(event.utterance_hypothesis, expected)
        self.assertIn("첫 번째 소식은 이미 전했습니다.", event.utterance_hypothesis)

    def test_sentence_head_revision_accepts_zero_confirmed_prefix(self) -> None:
        previous = "오후부터는 일부 지역에 구름이 많아졌습니다."
        current = "구름이 많아질 것으로 예상됩니다."
        expected = "오후부터는 일부 지역에 구름이 많아질 것으로 예상됩니다."

        revision = reconcile_adjacent_sentence_head_revision(
            previous, previous, current, 6, 0
        )
        self.assertIsNotNone(revision)
        self.assertEqual(revision[0], expected)

        assembler = SubtitleAssembler(0, enable_sentence_lifecycle=True)
        first = assembler.process(0, previous)
        self.assertEqual(first.confirmed_prefix_length, 0)
        event = assembler.process(1, current, shared_speech=True)
        self.assertEqual(event.match_type, "sentence_head_revision")
        self.assertEqual(event.rollover_text, "")
        self.assertEqual(event.utterance_hypothesis, expected)

    def test_sentence_head_revision_respects_confirmed_prefix_around_seam(self) -> None:
        current = "구름이 많아질 것으로 예상됩니다."

        before = SubtitleAssembler(0, enable_sentence_lifecycle=True)
        before.process(0, "먼저 안내드리며 오후부터는")
        established_before = before.process(
            1,
            "먼저 안내드리며 오후부터는 일부 지역에 구름이 많아졌습니다.",
            shared_speech=True,
        )
        before_seam = normalize_for_matching(
            established_before.utterance_hypothesis
        ).index(normalize_for_matching("구름이 많아졌습니다."))
        self.assertLess(established_before.confirmed_prefix_length, before_seam)
        direct_before = reconcile_adjacent_sentence_head_revision(
            established_before.utterance_hypothesis,
            established_before.raw,
            current,
            6,
            established_before.confirmed_prefix_length,
        )
        self.assertIsNotNone(direct_before)
        revised = before.process(2, current, shared_speech=True)
        self.assertEqual(revised.match_type, "sentence_head_revision")
        self.assertEqual(revised.rollover_text, "")

        after = SubtitleAssembler(0, enable_sentence_lifecycle=True)
        after.process(0, "오후부터는 일부 지역에 구름이 많아")
        established_after = after.process(
            1,
            "오후부터는 일부 지역에 구름이 많아졌습니다.",
            shared_speech=True,
        )
        after_seam = normalize_for_matching(
            established_after.utterance_hypothesis
        ).index(normalize_for_matching("구름이 많아졌습니다."))
        self.assertGreater(established_after.confirmed_prefix_length, after_seam)
        direct_after = reconcile_adjacent_sentence_head_revision(
            established_after.utterance_hypothesis,
            established_after.raw,
            current,
            6,
            established_after.confirmed_prefix_length,
        )
        self.assertIsNone(direct_after)
        separated = after.process(2, current, shared_speech=True)
        self.assertNotEqual(separated.match_type, "sentence_head_revision")
        self.assertEqual(
            separated.rollover_text,
            "오후부터는 일부 지역에 구름이 많아졌습니다.",
        )
        self.assertEqual(separated.utterance_hypothesis, current)

    def test_sentence_head_revision_preserves_active_text_before_previous_window(self) -> None:
        active = "한편 전문가들은 갑작스러운 기운 변화에 대비해 건강을 받았다."
        previous = "인가들은 갑작스러운 기운 변화에 대비해 건강을 받았다."
        current = "비해 건강관리에 주의할 것을 당부했습니다."
        expected = (
            "한편 전문가들은 갑작스러운 기운 변화에 대비해 "
            "건강관리에 주의할 것을 당부했습니다."
        )

        revision = reconcile_adjacent_sentence_head_revision(
            active, previous, current, 6, 0
        )
        self.assertIsNotNone(revision)
        self.assertEqual(revision[0], expected)

        assembler = SubtitleAssembler(0, enable_sentence_lifecycle=True)
        assembler.process(0, "한편 전문가들은 갑작스러운")
        assembler.process(1, previous, shared_speech=True)
        event = assembler.process(2, current, shared_speech=True)
        self.assertEqual(event.match_type, "sentence_head_revision")
        self.assertEqual(event.utterance_hypothesis, expected)
        self.assertTrue(event.utterance_hypothesis.startswith("한편 전문가들은"))

    def test_actual_new_sentence_is_not_sentence_head_revision(self) -> None:
        previous = "오늘 예산안을 검토했다."
        current = "예산안을 다시 설명합니다."

        self.assertIsNone(
            reconcile_adjacent_sentence_head_revision(
                previous, previous, current, 6, 0
            )
        )

        harness = GeneralSubtitleHarness()
        harness.speech(0, previous)
        assembly, events = harness.speech(1, current, shared_speech=True)
        self.assertNotEqual(assembly.match_type, "sentence_head_revision")
        self.assertEqual([event.status for event in events], ["final", "partial"])
        self.assertEqual(events[0].text, previous)
        self.assertEqual(events[1].publication_text, current)

    def test_completed_sentence_tail_without_punctuation_is_not_replayed(self) -> None:
        harness = GeneralSubtitleHarness()
        first = "한편 전문가들은 건강관리에 주의할 것을 당부했습니다."
        harness.speech(0, first)

        assembly, events = harness.speech(
            1,
            "했습니다 그럼요 외출하기 전에 날씨를 확인하세요.",
            shared_speech=True,
        )

        self.assertEqual(assembly.match_type, "sentence_tail_exact")
        self.assertEqual([event.status for event in events], ["final", "partial"])
        self.assertEqual(events[0].text, first)
        self.assertEqual(events[1].utterance_id, 2)
        self.assertEqual(
            events[1].text,
            "그럼요 외출하기 전에 날씨를 확인하세요.",
        )
        self.assertNotIn("했습니다 그럼요", events[1].text)

        _, continuation = harness.speech(
            2,
            "외출하기 전에 날씨를 확인하는 것도 좋은 방법입니다.",
            shared_speech=True,
        )
        self.assertEqual(continuation[-1].utterance_id, 2)
        self.assertTrue(continuation[-1].text.startswith("그럼요 "))
        self.assertNotIn("했습니다 그럼요", continuation[-1].text)

    def test_changed_sentence_ending_is_not_treated_as_replayed_tail(self) -> None:
        harness = GeneralSubtitleHarness()
        first = "오늘 점검을 완료했습니다."
        harness.speech(0, first)

        assembly, events = harness.speech(
            1,
            "했습니다만 추가 확인이 필요합니다.",
            shared_speech=True,
        )

        self.assertNotEqual(assembly.match_type, "sentence_tail_exact")
        self.assertEqual(events[0].status, "final")
        self.assertEqual(events[1].text, "했습니다만 추가 확인이 필요합니다.")

    def test_sentence_tail_words_are_preserved_without_shared_audio(self) -> None:
        harness = GeneralSubtitleHarness()
        harness.speech(0, "첫 발표를 완료했습니다.")

        assembly, events = harness.speech(
            1,
            "했습니다 다음 발표에서도 실제로 반복한 말입니다.",
            shared_speech=False,
        )

        self.assertNotEqual(assembly.match_type, "sentence_tail_exact")
        self.assertEqual(events[1].utterance_id, 2)
        self.assertEqual(
            events[1].text,
            "했습니다 다음 발표에서도 실제로 반복한 말입니다.",
        )

    def test_overlap_mode_keeps_existing_assembler_behavior(self) -> None:
        assembler = SubtitleAssembler(
            0,
            allow_shifted_head=False,
            enable_sentence_lifecycle=False,
        )
        first = "첫 번째 독립 문장입니다."
        second = "두 번째 독립 문장입니다."
        assembler.process(0, first)
        event = assembler.process(1, second, shared_speech=False)

        self.assertEqual(event.rollover_text, "")
        self.assertEqual(event.utterance_hypothesis, f"{first} {second}")


if __name__ == "__main__":
    unittest.main()
