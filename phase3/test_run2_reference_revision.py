"""Run2 W9/10 correction, with user-supplied reference (2026-10-27)."""
import unittest

from subtitle_assembler import SpeakerSubtitleState, SubtitleAssembler, normalize_for_matching


class Run2ReferenceRevisionTests(unittest.TestCase):
    previous = "시민들은 대중교통을 이용하여"
    current = "대중교통을 이용하거나 도로 상황을 확인하며 이동"
    expected = "시민들은 대중교통을 이용하거나 도로 상황을 확인하며 이동"

    def test_recorded_supported_terminal_revision_reaches_partial_and_final(self):
        assembler = SubtitleAssembler(0, enable_sentence_lifecycle=True)
        state = SpeakerSubtitleState(0, require_final_support=True)
        for window, raw in ((9, self.previous), (10, self.current)):
            event = assembler.process(window, raw, shared_speech=True, supported_tail_revision=True)
            partial = state.process(window, event.utterance_hypothesis, True, 2 * window + 3,
                                    confirmed_prefix_length=event.confirmed_prefix_length,
                                    source_supported=True, source_text=raw)[0]
        self.assertEqual(event.match_type, "fuzzy_replace")
        self.assertEqual(partial.publication_text, self.expected)
        self.assertEqual(state.flush(10, 23).publication_text, self.expected)

    def test_terminal_revision_requires_adjacent_shared_and_supported_audio(self):
        for window, shared, supported in ((10, True, False), (10, False, True),
                                          (10, None, True), (11, True, True)):
            with self.subTest(window=window, shared=shared, supported=supported):
                assembler = SubtitleAssembler(0)
                assembler.process(9, self.previous)
                event = assembler.process(window, self.current, shared_speech=shared,
                                          supported_tail_revision=supported)
                self.assertIn(self.previous, event.utterance_hypothesis)
                self.assertNotEqual(event.match_type, "fuzzy_replace")

    def test_confirmed_prefix_and_final_history_are_preserved(self):
        assembler = SubtitleAssembler(0)
        assembler.process(0, "먼저 확정한 소식입니다.")
        assembler.reset_utterance(final_text="먼저 확정한 소식입니다.")
        assembler.process(8, self.previous)
        confirmed = assembler.process(9, self.previous, shared_speech=True)
        event = assembler.process(10, self.current, shared_speech=True, supported_tail_revision=True)
        self.assertEqual(event.confirmed_prefix_length, confirmed.confirmed_prefix_length)
        self.assertTrue(normalize_for_matching(event.utterance_hypothesis).startswith(
            normalize_for_matching(self.previous)))
        self.assertTrue(event.session_text.startswith("먼저 확정한 소식입니다."))

    def test_supported_noise_in_an_earlier_word_is_not_promoted(self):
        assembler = SubtitleAssembler(0)
        assembler.process(9, "오후부터는 구름이 많아질 것으로")
        event = assembler.process(10, "유름이 많아질 것으로 예상됩니다.",
                                  shared_speech=True, supported_tail_revision=True)
        self.assertIn("구름이 많아질 것으로 예상됩니다.", event.utterance_hypothesis)
        self.assertNotIn("유름이", event.utterance_hypothesis)

    def test_terminal_revision_preserves_confirmed_words_before_the_seam(self):
        assembler = SubtitleAssembler(0)
        assembler.process(7, "시민들은")
        assembler.process(8, "시민들은", shared_speech=True)
        assembler.process(9, self.previous, shared_speech=True)
        event = assembler.process(10, self.current, shared_speech=True, supported_tail_revision=True)
        self.assertEqual(event.utterance_hypothesis, self.expected)
        self.assertEqual(event.confirmed_prefix_length, len(normalize_for_matching("시민들은")))

    def test_terminal_revision_is_not_specific_to_the_recorded_words(self):
        assembler = SubtitleAssembler(0)
        assembler.process(0, "주민들은 대중교통을 이용하여")
        event = assembler.process(1, "대중교통을 이용하면서 주변 상황을 확인합니다",
                                  shared_speech=True, supported_tail_revision=True)
        self.assertEqual(event.utterance_hypothesis,
                         "주민들은 대중교통을 이용하면서 주변 상황을 확인합니다")


if __name__ == "__main__":
    unittest.main()
