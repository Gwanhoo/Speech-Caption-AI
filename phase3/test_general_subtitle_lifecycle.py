from __future__ import annotations

import unittest

from subtitle_assembler import SpeakerSubtitleState, SubtitleAssembler


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
