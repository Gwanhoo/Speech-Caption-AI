from __future__ import annotations

import unittest

from subtitle_assembler import SpeakerSubtitleState, SubtitleAssembler, normalize_for_matching


def production_source_fragment(event) -> str:
    if (
        event.match_type in {"fuzzy_replace", "supported_tail_replace"}
        or not event.new_fragment
    ):
        return event.raw_fragment
    return event.new_fragment


class ProductionEvidenceRegressions(unittest.TestCase):
    def test_stable_fuzzy_overlap_keeps_existing_word_and_appends_new_suffix(self) -> None:
        assembler = SubtitleAssembler(0)
        assembler.process(4, "오후부터는 일부 지역에 구름이 많아질 것으로")

        event = assembler.process(5, "유름이 많아질 것으로 예상됩니다. 시민들은")

        self.assertEqual(event.match_type, "fuzzy")
        self.assertEqual(event.new, "예상됩니다. 시민들은")
        self.assertIn("구름이 많아질 것으로 예상됩니다. 시민들은", event.utterance_hypothesis)
        self.assertNotIn("유름이", event.utterance_hypothesis)

    def test_normal_exact_overlap_still_appends_new_text(self) -> None:
        assembler = SubtitleAssembler(0)
        assembler.process(0, "시민들은 대중교통을 이용하거나")

        event = assembler.process(1, "대중교통을 이용하거나 도로 상황을 확인하며")

        self.assertEqual(event.match_type, "token_exact")
        self.assertEqual(event.new, "도로 상황을 확인하며")
        self.assertEqual(
            event.utterance_hypothesis,
            "시민들은 대중교통을 이용하거나 도로 상황을 확인하며",
        )

    def test_duplicate_only_input_creates_no_new_fragment_or_hypothesis(self) -> None:
        assembler = SubtitleAssembler(0)
        first = assembler.process(0, "오늘 준비한 소식은 여기까지입니다.")

        duplicate = assembler.process(1, "오늘 준비한 소식은 여기까지입니다.")

        self.assertTrue(duplicate.duplicate_only)
        self.assertEqual(duplicate.new, "")
        self.assertEqual(duplicate.utterance_hypothesis, first.utterance_hypothesis)

    def test_pipeline_end_preserves_every_admitted_source_supported_suffix(self) -> None:
        assembler = SubtitleAssembler(0)
        state = SpeakerSubtitleState(0, require_final_support=True)
        fragments = [
            "시민들은 대중교통을 이용하거나 도로상환",
            "하거나 도로 상황을 확인하며 이동하고 있습니다.",
            "네, 현재까지 큰 교통입니다.",
            "시작까지 큰 교통 혼잡은 발생하지 않았습니다.",
            "않았습니다. 한편 전문가들은 갑작스러워",
            "인가들은 갑작스러운 기운 변화에 대비해 건강",
            "비해 건강관리에 주의할 것을 당부했습니다.",
            "당부했습니다. 그럼요 외출하기",
        ]
        last_event = None
        for window, fragment in enumerate(fragments):
            assembly = assembler.process(window, fragment)
            last_event = state.process(
                window,
                assembly.utterance_hypothesis,
                True,
                3 + 2 * window,
                confirmed_prefix_length=assembly.confirmed_prefix_length,
                source_supported=True,
                source_text=production_source_fragment(assembly),
            )[0]

        self.assertIsNotNone(last_event)
        self.assertEqual(last_event.publication_text, last_event.text)
        final = state.flush(len(fragments) - 1, 3 + 2 * (len(fragments) - 1))
        self.assertIsNotNone(final)
        self.assertEqual(final.stability_action, "tentative_retained_at_final")
        self.assertEqual(final.text, last_event.text)
        self.assertTrue(final.text.endswith("당부했습니다. 그럼요 외출하기"))
        self.assertNotIn("도로상환 황을", final.text)

    def test_unsupported_or_non_primary_text_is_not_revived_at_pipeline_end(self) -> None:
        state = SpeakerSubtitleState(0, require_final_support=True)
        accepted = "선택된 primary의 검증된 자막"
        rejected = "거부되었거나 non-primary인 자막"
        state.process(
            0,
            accepted,
            True,
            3,
            confirmed_prefix_length=0,
            source_supported=True,
            source_text=accepted,
        )
        state.process(
            1,
            accepted + " " + rejected,
            True,
            5,
            confirmed_prefix_length=len(normalize_for_matching(accepted)),
            source_supported=False,
            source_text="",
        )

        final = state.flush(1, 5)

        self.assertIsNotNone(final)
        self.assertEqual(final.text, accepted)
        self.assertNotIn(rejected, final.text)


if __name__ == "__main__":
    unittest.main()
