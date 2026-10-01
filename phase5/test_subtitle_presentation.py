from __future__ import annotations

import unittest

from phase5.subtitle_presentation import SubtitlePresentationState


def event(
    speaker: str,
    utterance_id: int,
    sequence: int,
    text: str,
    status: str = "partial",
) -> dict[str, object]:
    return {
        "speaker": speaker,
        "utterance_id": utterance_id,
        "sequence": sequence,
        "status": status,
        "text": text,
    }


class SubtitlePresentationStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = SubtitlePresentationState()

    def texts(self) -> list[str]:
        return [entry.text for entry in self.state.entries]

    def test_partial_updates_one_current_caption(self) -> None:
        self.assertTrue(self.state.apply(event("speaker_0", 1, 1, "안녕하세요")))
        self.assertTrue(self.state.apply(event("speaker_0", 1, 2, "안녕하세요 오늘")))
        self.assertTrue(self.state.apply(event("speaker_0", 1, 3, "안녕하세요 오늘 회의")))
        self.assertEqual(self.texts(), ["안녕하세요 오늘 회의"])
        self.assertEqual(self.state.entries[0].status, "partial")

    def test_partial_becomes_one_final_caption(self) -> None:
        self.state.apply(event("speaker_0", 1, 1, "안녕하세요 오늘 회의"))
        self.assertTrue(
            self.state.apply(event("speaker_0", 1, 2, "안녕하세요 오늘 회의", "final"))
        )
        self.assertEqual(self.texts(), ["안녕하세요 오늘 회의"])
        self.assertEqual(self.state.entries[0].status, "final")

    def test_new_utterance_follows_final_caption(self) -> None:
        self.state.apply(event("speaker_0", 1, 1, "안녕하세요 오늘 회의", "final"))
        self.state.apply(event("speaker_1", 1, 2, "다음 문장"))
        self.assertEqual(self.texts(), ["안녕하세요 오늘 회의", "다음 문장"])
        self.assertEqual([entry.status for entry in self.state.entries], ["final", "partial"])

    def test_duplicate_final_event_is_ignored_but_later_same_words_are_allowed(self) -> None:
        final = event("speaker_0", 1, 1, "같은 문장", "final")
        self.assertTrue(self.state.apply(final))
        self.assertFalse(self.state.apply(final))
        self.assertTrue(self.state.apply(event("speaker_0", 2, 2, "같은 문장", "final")))
        self.assertEqual(self.texts(), ["같은 문장", "같은 문장"])

    def test_out_of_order_event_cannot_replace_the_current_caption(self) -> None:
        self.state.apply(event("speaker_0", 1, 2, "최신 자막"))
        self.assertFalse(self.state.apply(event("speaker_0", 1, 1, "지연된 자막")))
        self.assertEqual(self.texts(), ["최신 자막"])

    def test_two_partials_update_and_finalize_independently(self) -> None:
        self.state.apply(event("speaker_0", 1, 1, "오늘 학교에 갔는데"))
        self.state.apply(event("speaker_1", 1, 2, "그러면 다음 단계는"))
        self.assertEqual(self.texts(), ["오늘 학교에 갔는데", "그러면 다음 단계는"])
        self.state.apply(event("speaker_0", 1, 3, "오늘 학교에 갔는데 친구가"))
        self.assertEqual(self.texts(), ["오늘 학교에 갔는데 친구가", "그러면 다음 단계는"])
        self.assertFalse(self.state.apply(event("speaker_0", 1, 1, "옛 자막")))
        self.state.apply(event("speaker_0", 1, 4, "오늘 학교에 갔는데 친구가", "final"))
        self.assertEqual([entry.status for entry in self.state.entries], ["final", "partial"])
        self.assertEqual(self.state.entries[-1].speaker_id, "speaker_1")
        self.assertFalse(self.state.apply(event("speaker_0", 1, 5, "재개 금지")))

    def test_active_partials_take_priority_over_recent_finals(self) -> None:
        self.state.apply(event("speaker_0", 1, 1, "첫 확정", "final"))
        self.state.apply(event("speaker_1", 1, 2, "둘째 확정", "final"))
        self.state.apply(event("speaker_0", 2, 3, "진행 0"))
        self.state.apply(event("speaker_1", 2, 4, "진행 1"))
        self.assertEqual(self.texts(), ["둘째 확정", "진행 0", "진행 1"])
        self.assertEqual(len(self.state.entries), 3)
        self.state.apply(event("speaker_0", 2, 5, "완료 0", "final"))
        self.assertEqual(self.texts(), ["둘째 확정", "완료 0", "진행 1"])

    def test_sequences_are_checked_per_stream_and_clear_resets_both(self) -> None:
        self.state.apply(event("speaker_1", 1, 20, "진행 1"))
        self.assertTrue(self.state.apply(event("speaker_0", 1, 10, "진행 0")))
        self.assertEqual(self.texts(), ["진행 0", "진행 1"])
        self.assertFalse(self.state.apply(event("speaker_1", 1, 19, "옛 확정", "final")))
        self.state.clear()
        self.assertEqual(self.state.entries, ())
        self.assertTrue(self.state.apply(event("speaker_0", 1, 1, "재시작")))

    def test_long_text_is_preserved_without_inserted_line_breaks(self) -> None:
        text = "이 문장은 화면의 폭에 따라 Qt가 자연스럽게 줄바꿈해야 하며 데이터에는 임의의 줄바꿈이 들어가면 안 됩니다."
        self.state.apply(event("speaker_0", 1, 1, text))
        self.assertEqual(self.state.entries[0].text, text)
        self.assertNotIn("\n", self.state.entries[0].text)

    def test_blank_text_is_ignored(self) -> None:
        for text in ("", " ", "\t\n"):
            self.assertFalse(self.state.apply(event("speaker_0", 1, 1, text)))
        self.assertEqual(self.state.entries, ())

    def test_final_history_is_bounded(self) -> None:
        for sequence in range(1, 6):
            self.assertTrue(
                self.state.apply(event("speaker_0", sequence, sequence, f"문장 {sequence}", "final"))
            )
        self.assertEqual(self.texts(), ["문장 4", "문장 5"])
        self.assertEqual(len(self.state.entries), self.state.MAX_FINAL_ENTRIES)


if __name__ == "__main__":
    unittest.main()
