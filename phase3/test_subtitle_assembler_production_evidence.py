from __future__ import annotations

import json
import unittest
from pathlib import Path

from subtitle_assembler import (
    SpeakerSubtitleState, SubtitleAssembler, complete_fuzzy_overlap_token, fuzzy_suffix_prefix,
    normalize_for_matching,
)


LOGGED_WINDOW_FIXTURE = (
    Path(__file__).with_name("test_fixtures") / "gui_last_run_overlap_windows.json"
)


def production_source_fragment(event) -> str:
    if (
        event.match_type in {"fuzzy_replace", "supported_tail_replace"}
        or not event.new_fragment
    ):
        return event.raw_fragment
    return event.new_fragment


def logged_windows() -> dict[int, dict[str, object]]:
    """Load the minimal, immutable primary-stream extract from gui_last_run."""
    fixture = json.loads(LOGGED_WINDOW_FIXTURE.read_text(encoding="utf-8"))
    return {entry["window"]: entry for entry in fixture["windows"]}


class LoggedWindowReplay:
    """Exercise assembler/state ordering with recorded text, VAD, and timing."""

    def __init__(self) -> None:
        self.assembler = SubtitleAssembler(0, enable_sentence_lifecycle=True)
        self.state = SpeakerSubtitleState(0, require_final_support=True)

    def process(
        self,
        record: dict[str, object],
        *,
        supported_tail_revision: bool | None = None,
    ):
        overlap = record["overlap"]
        assembly = self.assembler.process(
            record["window"],
            record["raw"],
            shared_speech=overlap.get("shared_speech"),
            supported_tail_revision=(
                overlap.get("supported_tail_revision", False)
                if supported_tail_revision is None
                else supported_tail_revision
            ),
        )
        events = []
        if assembly.rollover_text and self.state.partial_text:
            final = self.state.finalize(
                record["window"],
                record["stream_end_seconds"],
                assembly.rollover_reason or "",
            )
            self.assembler.commit_finalized_text(final.text)
            events.append(final)
        events.extend(
            self.state.process(
                record["window"],
                assembly.utterance_hypothesis,
                record["vad"]["speech_detected"],
                record["stream_end_seconds"],
                confirmed_prefix_length=assembly.confirmed_prefix_length,
                source_supported=bool(record["raw"]),
                source_text=production_source_fragment(assembly),
            )
        )
        return assembly, events


class NewsFuzzyEndingRegressions(unittest.TestCase):
    def test_recorded_window_9_preserves_ending_through_assembly_and_publication(self) -> None:
        fixture = json.loads((LOGGED_WINDOW_FIXTURE.parent /
            "news_20261022_160827_windows.json").read_text(encoding="utf-8"))
        records = {row["window"]: row for row in fixture["windows"]}
        previous = records[8]
        current = records[9]
        self.assertEqual(previous["raw_stt"], "인증 없이 외부에서 임의로 코드를 실행...")
        self.assertEqual(current["raw_stt"], "의미로 코드를 실행하거나 시스템을 장악할 수")
        self.assertEqual(current["raw"], current["raw_stt"])
        score, cut = fuzzy_suffix_prefix(previous["raw"], current["raw"], 6)
        self.assertAlmostEqual(score, 14 / 15)
        self.assertEqual(cut, 10)
        self.assertEqual(current["raw"][:cut], "의미로 코드를 실행")
        self.assertEqual(current["logged_assembly"]["overlap"], "의미로 코드를 실행하거나")

        replay = LoggedWindowReplay()
        for window in range(9):
            assembly, _ = replay.process(records[window])
            self.assertEqual(assembly.utterance_hypothesis,
                             records[window]["logged_assembly"]["utterance_hypothesis"])
        prefix = assembly.utterance_hypothesis.removesuffix("실행...")
        event, partials = replay.process(current)
        expected = prefix + "실행하거나 시스템을 장악할 수"
        self.assertEqual(event.previous, previous["raw"])
        self.assertEqual(event.raw, current["raw_stt"])
        self.assertEqual(event.match_type, "fuzzy")
        self.assertEqual(event.overlap, current["raw"][:cut])
        self.assertEqual(event.new_fragment, current["raw"][cut:])
        self.assertEqual(event.utterance_hypothesis, expected)
        self.assertEqual(event.session_text, expected)
        self.assertEqual(event.confirmed_prefix_length, assembly.confirmed_prefix_length)
        self.assertEqual(event.rollover_text, "")
        self.assertFalse(event.duplicate_only)
        self.assertEqual(partials[-1].publication_text, expected)

        following, partials = replay.process(records[10])
        expected += " 있는 결함으로 알려졌습니다."
        self.assertEqual(following.match_type, "token_exact")
        self.assertEqual(following.utterance_hypothesis, expected)
        self.assertEqual(partials[-1].publication_text, expected)
        self.assertEqual(replay.state.flush(10, 23).publication_text, expected)

    def test_exact_terminal_stem_extension_is_not_duplicated_or_split(self) -> None:
        # Synthetic variants, separate from the recorded news fixture above.
        for ending in ("하거나", "하면서", "하고", "하면"):
            for punctuation in ("", "...", "."):
                with self.subTest(ending=ending, punctuation=punctuation):
                    assembler = SubtitleAssembler(0, enable_sentence_lifecycle=True)
                    assembler.process(8, "인증 없이 외부에서 임의로 코드를 실행" + punctuation)
                    raw = "의미로 코드를 실행" + ending + " 시스템을 확인합니다"
                    event = assembler.process(9, raw, shared_speech=True)
                    expected = "인증 없이 외부에서 임의로 코드를 실행" + ending + " 시스템을 확인합니다"
                    self.assertEqual(event.utterance_hypothesis, expected)
                    self.assertEqual(event.new_fragment, ending + " 시스템을 확인합니다")
                    duplicate = assembler.process(10, raw, shared_speech=True)
                    self.assertTrue(duplicate.duplicate_only)
                    self.assertEqual(duplicate.new_fragment, "")
                    self.assertEqual(duplicate.utterance_hypothesis, expected)

    def test_gap_or_no_shared_speech_does_not_join_stems(self) -> None:
        previous = "인증 없이 외부에서 임의로 코드를 실행"
        raw = "의미로 코드를 실행하거나 시스템을 장악할 수"
        for window, shared in ((10, True), (9, False)):
            with self.subTest(window=window, shared=shared):
                assembler = SubtitleAssembler(0)
                assembler.process(8, previous)
                event = assembler.process(window, raw, shared_speech=shared)
                self.assertEqual(event.match_type, "none")
                self.assertEqual(event.new_fragment, raw)
                self.assertEqual(event.utterance_hypothesis, previous + " " + raw)

    def test_changed_or_unobserved_terminal_token_keeps_boundary_guard(self) -> None:
        # These are synthetic boundary guards, not additional news observations.
        for previous, active, current, cut, expected in (
            ("도로상환", "도로상환", "도로 상황을 확인하며", 4, 6),
            ("실행", "실현", "실행하거나", 2, 5),
            ("실현", "실행", "실행하거나", 2, 5),
            ("재실행", "재실행", "실행하거나", 2, 5),
            ("실행", "실행", "실행A", 2, 3),
            ("하", "하", "하면서", 1, 3),
        ):
            with self.subTest(previous=previous, active=active, current=current):
                self.assertEqual(complete_fuzzy_overlap_token(
                    current, cut, previous=previous, active=active), expected)

        assembler = SubtitleAssembler(0)
        previous = "시민들은 대중교통을 이용하거나 도로상환"
        assembler.process(8, previous)
        event = assembler.process(9, "하거나 도로 상황을 확인하며 이동하고 있습니다.")
        self.assertEqual(event.match_type, "fuzzy")
        self.assertEqual(event.new_fragment, "확인하며 이동하고 있습니다.")
        self.assertEqual(event.utterance_hypothesis,
                         previous + " 확인하며 이동하고 있습니다.")

    def test_stem_extension_preserves_independently_confirmed_prefix(self) -> None:
        assembler = SubtitleAssembler(0)
        previous = "인증 없이 외부에서 임의로 코드를 실행..."
        assembler.process(7, previous)
        confirmed = assembler.process(8, previous)
        self.assertGreater(confirmed.confirmed_prefix_length, 0)
        event = assembler.process(9, "의미로 코드를 실행하거나 시스템을 장악할 수")
        self.assertEqual(event.utterance_hypothesis,
                         "인증 없이 외부에서 임의로 코드를 실행하거나 시스템을 장악할 수")
        self.assertEqual(event.confirmed_prefix_length, confirmed.confirmed_prefix_length)


class FuzzyContinuationSafetyRegressions(unittest.TestCase):
    def test_confirmed_stem_keeps_raw_word_boundary_through_publication_and_final(self) -> None:
        # Synthetic safety cases. The recorded news fixture is unchanged.
        previous = "외부에서 임의로 코드를 실행"
        for suffix in ("하거나", "하면서", "파일을", "중지를"):
            for punctuation in ("", "...", ",", "?", "…"):
                for confirmed in ("외부에서 임의로 코드를", previous[:-1], previous):
                    with self.subTest(suffix=suffix, punctuation=punctuation, confirmed=confirmed):
                        assembler = SubtitleAssembler(0, enable_sentence_lifecycle=True)
                        state = SpeakerSubtitleState(0, require_final_support=True)
                        raw = "의미로 코드를 실행" + suffix + " 확인합니다"
                        for window, text in enumerate((confirmed, previous + punctuation, raw)):
                            event = assembler.process(window, text, shared_speech=True)
                            partial = state.process(
                                window, event.utterance_hypothesis, True, 3 + 2 * window,
                                confirmed_prefix_length=event.confirmed_prefix_length,
                                source_supported=True, source_text=production_source_fragment(event),
                            )[0]
                            if window == 1:
                                protected = event.confirmed_prefix_length
                                protected_text = normalize_for_matching(event.utterance_hypothesis)[:protected]
                        expected = previous + suffix + " 확인합니다"
                        self.assertGreater(protected, 0)
                        self.assertEqual(event.match_type, "fuzzy")
                        self.assertEqual(event.utterance_hypothesis, expected)
                        self.assertEqual(event.confirmed_prefix_length, protected)
                        self.assertEqual(normalize_for_matching(partial.stable_text), protected_text)
                        self.assertEqual(partial.text, expected)
                        self.assertEqual(partial.publication_text, expected)
                        self.assertEqual(event.rollover_text, "")
                        final = state.flush(2, 7)
                        self.assertEqual(final.text, expected)
                        self.assertEqual(final.publication_text, expected)

    def test_distinct_raw_and_active_tail_never_uses_fuzzy_continuation(self) -> None:
        # A prior approximate overlap can leave the old spelling in active text.
        assembler = SubtitleAssembler(0)
        assembler.process(0, "시민들은 대중교통을 이용하거나 도로상환")
        mismatch = assembler.process(1, "이용하거나 도로 상황을")
        self.assertEqual(mismatch.match_type, "fuzzy")
        self.assertEqual(assembler.previous, "이용하거나 도로 상황을")
        self.assertTrue(assembler.utterance_hypothesis.endswith("도로상환"))
        event = assembler.process(2, "이용하거니 도로 상황을보고 이동합니다")
        self.assertEqual(event.match_type, "fuzzy")
        self.assertEqual(event.new_fragment, "이동합니다")
        self.assertEqual(event.utterance_hypothesis,
                         "시민들은 대중교통을 이용하거나 도로상환 이동합니다")

    def test_spaces_and_punctuation_remain_real_word_boundaries(self) -> None:
        previous = "외부에서 임의로 코드를 실행"
        for separator in (" ", "  ", "\t", "\n", ", ", ". "):
            with self.subTest(separator=separator):
                assembler = SubtitleAssembler(0)
                assembler.process(0, previous)
                event = assembler.process(1, "의미로 코드를 실행" + separator + "파일을 확인합니다")
                self.assertEqual(event.match_type, "fuzzy")
                self.assertEqual(event.new_fragment, "파일을 확인합니다")
                self.assertEqual(event.utterance_hypothesis, previous + " 파일을 확인합니다")

    def test_continuous_extensions_and_duplicate_keep_one_supported_word(self) -> None:
        assembler = SubtitleAssembler(0)
        state = SpeakerSubtitleState(0, require_final_support=True)
        previous = "외부에서 임의로 코드를 실행"
        raws = (previous, previous, "의미로 코드를 실행하거나",
                "의미로 코드를 실행하거나 시스템을 확인합니다",
                "의미로 코드를 실행하거나 시스템을 확인합니다")
        for window, raw in enumerate(raws):
            event = assembler.process(window, raw, shared_speech=True)
            partial = state.process(
                window, event.utterance_hypothesis, True, 3 + 2 * window,
                confirmed_prefix_length=event.confirmed_prefix_length,
                source_supported=True, source_text=production_source_fragment(event),
            )[0]
            self.assertEqual(partial.publication_text, event.utterance_hypothesis)
        expected = previous + "하거나 시스템을 확인합니다"
        self.assertTrue(event.duplicate_only)
        self.assertEqual(event.new_fragment, "")
        self.assertEqual(event.utterance_hypothesis, expected)
        self.assertEqual(state.flush(4, 11).publication_text, expected)

    def test_unsupported_extension_is_not_promoted_or_published(self) -> None:
        assembler = SubtitleAssembler(0)
        state = SpeakerSubtitleState(0, require_final_support=True)
        previous = "외부에서 임의로 코드를 실행"
        for window, raw in enumerate((previous, previous, "의미로 코드를 실행하거나")):
            event = assembler.process(window, raw, shared_speech=True)
            partial = state.process(
                window, event.utterance_hypothesis, True, 3 + 2 * window,
                confirmed_prefix_length=event.confirmed_prefix_length,
                source_supported=window < 2, source_text=production_source_fragment(event),
            )[0]
        self.assertEqual(event.utterance_hypothesis, previous + "하거나")
        self.assertEqual(partial.stable_text, previous)
        self.assertEqual(partial.tentative_text, "하거나")
        self.assertEqual(partial.text, previous + "하거나")
        self.assertEqual(partial.publication_text, previous)
        self.assertEqual(state.flush(2, 7).publication_text, previous)

    def test_finalized_session_and_rejected_window_order_are_unchanged(self) -> None:
        assembler = SubtitleAssembler(0)
        finalized = "앞서 확정한 안내입니다."
        assembler.process(0, finalized)
        assembler.reset_utterance(final_text=finalized)
        previous = "외부에서 임의로 코드를 실행"
        assembler.process(1, previous)
        raw = "의미로 코드를 실행하거나 시스템을 확인합니다"
        event = assembler.process(2, raw, shared_speech=True)
        expected = previous + "하거나 시스템을 확인합니다"
        self.assertEqual(event.utterance_hypothesis, expected)
        self.assertEqual(event.session_text, finalized + " " + expected)
        for window in (2, 1):
            with self.subTest(window=window), self.assertRaises(ValueError):
                assembler.process(window, raw)
            self.assertEqual(assembler.utterance_hypothesis, expected)
            self.assertEqual(assembler.assembled, event.session_text)
            self.assertEqual(assembler.last_window, 2)


class ReplacementSafetyRegressions(unittest.TestCase):
    def test_recorded_deletion_only_revision_keeps_observed_words_and_new_suffix(self) -> None:
        fixture = json.loads((LOGGED_WINDOW_FIXTURE.parent /
            "news_20261022_160827_replacement_windows.json").read_text(encoding="utf-8"))
        records = {r["window"]: r for r in fixture["windows"]}
        self.assertEqual(records[11]["raw"], "이번 사태 파급력이 큰 건")
        self.assertEqual(records[12]["raw"], "사급력이 큰 건 PHP가 인터넷 정보...")
        self.assertEqual(records[12]["logged_assembly"]["match_type"], "fuzzy_replace")
        self.assertEqual(records[12]["logged_assembly"]["utterance_hypothesis"],
                         "이번 사급력이 큰 건 PHP가 인터넷 정보...")
        self.assertTrue(records[12]["overlap"]["supported_tail_revision"])
        self.assertEqual(records[12]["overlap"]["continuous_tail_coverage"], 0.998)
        replay = LoggedWindowReplay()
        replay.process(records[11])
        event, partials = replay.process(records[12])
        expected = records[11]["raw"] + " PHP가 인터넷 정보..."
        self.assertEqual(event.utterance_hypothesis, expected)
        self.assertEqual(event.match_type, "fuzzy")
        self.assertEqual(event.new_fragment, "PHP가 인터넷 정보...")
        self.assertEqual(partials[-1].publication_text, expected)
        _, following = replay.process(records[13])
        self.assertEqual(following[0].status, "final")
        self.assertEqual(following[0].publication_text, expected)
        self.assertEqual(following[-1].publication_text, records[13]["raw"])

    def test_deletion_only_revision_and_repetition_do_not_grant_support(self) -> None:
        # Synthetic support/duplicate variations of the recorded pair above.
        for support in (False, True):
            for confirmed in (False, True):
                with self.subTest(support=support, confirmed=confirmed):
                    assembler = SubtitleAssembler(0)
                    old = "이번 사태 파급력이 큰 건"
                    raw = "사급력이 큰 건 PHP가 인터넷 정보"
                    assembler.process(9, old)
                    if confirmed:
                        assembler.process(10, old)
                    first = assembler.process(11 if confirmed else 10, raw,
                        shared_speech=True, supported_tail_revision=support)
                    self.assertEqual(first.utterance_hypothesis, old + " PHP가 인터넷 정보")
                    self.assertEqual(first.confirmed_prefix_length,
                                     len(normalize_for_matching(old)) if confirmed else 0)
                    duplicate = assembler.process(first.window + 1, raw,
                        shared_speech=True, supported_tail_revision=support)
                    self.assertTrue(duplicate.duplicate_only)
                    self.assertEqual(duplicate.utterance_hypothesis, first.utterance_hypothesis)

    def test_supported_tail_revision_cannot_delete_text_before_previous_raw(self) -> None:
        assembler = SubtitleAssembler(0)
        older = "별도 앞부분 "
        previous = "제 손은 그냥과와 높아지는"
        assembler.process(0, older + previous)
        established = assembler.process(1, previous)
        self.assertEqual(established.confirmed_prefix_length, 0)
        raw = "분양가와 높아지는 청약문턱에 지친 세대가"
        event = assembler.process(2, raw, shared_speech=True, supported_tail_revision=True)
        self.assertEqual(event.match_type, "supported_tail_replace")
        self.assertEqual(event.utterance_hypothesis, older + raw)

    def test_confirmed_fuzzy_substitution_is_retained_but_new_speech_is_appended(self) -> None:
        assembler = SubtitleAssembler(0)
        old = "오늘은 익스트림 모두"
        assembler.process(0, old)
        confirmed = assembler.process(1, old)
        event = assembler.process(2, "익스트림 모드 실행합니다", shared_speech=True,
                                  supported_tail_revision=True)
        self.assertEqual(event.utterance_hypothesis, old + " 실행합니다")
        self.assertEqual(event.confirmed_prefix_length, confirmed.confirmed_prefix_length)
        self.assertEqual(event.match_type, "fuzzy")
        self.assertEqual(event.new_fragment, "실행합니다")

    def test_confirmed_tail_is_not_duplicated_by_supported_replacement(self) -> None:
        assembler = SubtitleAssembler(0)
        old = "네, 현재까지 큰 교육을"
        assembler.process(0, old)
        confirmed = assembler.process(1, old)
        event = assembler.process(2, "현재까지 큰 교통 혼잡은 발생하지 않았습니다.",
                                  shared_speech=True, supported_tail_revision=True)
        self.assertEqual(event.utterance_hypothesis, old + " 혼잡은 발생하지 않았습니다.")
        self.assertEqual(event.confirmed_prefix_length, confirmed.confirmed_prefix_length)
        self.assertEqual(event.match_type, "fuzzy")
        self.assertEqual(event.new_fragment, "혼잡은 발생하지 않았습니다.")

    def test_supported_substitution_of_unconfirmed_tail_still_reaches_final(self) -> None:
        for old, raw, match_type, expected in (
            ("제 손은 그냥과와 높아지는", "분양가와 높아지는 청약문턱에 지친 세대가",
             "supported_tail_replace", "분양가와 높아지는 청약문턱에 지친 세대가"),
            ("오늘은 익스트림 모두", "익스트림 모드 실행합니다",
             "fuzzy_replace", "오늘은 익스트림 모드 실행합니다"),
        ):
            with self.subTest(old=old):
                assembler = SubtitleAssembler(0)
                state = SpeakerSubtitleState(0, require_final_support=True)
                for window, text in enumerate((old, raw)):
                    event = assembler.process(window, text, shared_speech=True,
                                              supported_tail_revision=True)
                    partial = state.process(window, event.utterance_hypothesis, True,
                        3 + 2 * window, confirmed_prefix_length=event.confirmed_prefix_length,
                        source_supported=True, source_text=production_source_fragment(event))[0]
                self.assertEqual(event.match_type, match_type)
                self.assertEqual(partial.publication_text, expected)
                self.assertEqual(state.flush(1, 5).publication_text, expected)

    def test_matching_confirmed_anchor_still_allows_supported_tail_correction(self) -> None:
        for prefix in ("네, 현재까지", "네, 현재까지 큰 교"):
            with self.subTest(prefix=prefix):
                assembler = SubtitleAssembler(0)
                assembler.process(0, prefix)
                confirmed = assembler.process(1, "네, 현재까지 큰 교육을")
                raw = "현재까지 큰 교통 혼잡은 발생하지 않았습니다."
                event = assembler.process(2, raw, shared_speech=True,
                                          supported_tail_revision=True)
                self.assertEqual(event.utterance_hypothesis, "네, " + raw)
                self.assertEqual(event.match_type, "supported_tail_replace")
                self.assertEqual(event.confirmed_prefix_length, confirmed.confirmed_prefix_length)

    def test_supported_sentence_revision_cannot_lower_confirmed_prefix(self) -> None:
        assembler = SubtitleAssembler(0, enable_sentence_lifecycle=True)
        state = SpeakerSubtitleState(0, require_final_support=True)
        old = "오전에는 비교적 맑은 날씨가 이어졌습니다."
        raw = "날씨가 이어졌지만 오후부터는 일부 지역에"
        for window, text in enumerate((old[:-1], old, raw)):
            event = assembler.process(window, text, shared_speech=True,
                                      supported_tail_revision=True)
            partial = state.process(window, event.utterance_hypothesis, True,
                3 + 2 * window, confirmed_prefix_length=event.confirmed_prefix_length,
                source_supported=True, source_text=production_source_fragment(event))[0]
            if window == 1:
                protected = event.confirmed_prefix_length
        expected = old + " 오후부터는 일부 지역에"
        self.assertEqual(event.confirmed_prefix_length, protected)
        self.assertEqual(event.utterance_hypothesis, expected)
        self.assertEqual(partial.publication_text, expected)
        self.assertEqual(state.flush(2, 7).publication_text, expected)

    def test_retained_words_do_not_grant_acoustic_support_to_new_suffix(self) -> None:
        assembler = SubtitleAssembler(0)
        state = SpeakerSubtitleState(0, require_final_support=True)
        old = "이번 사태 파급력이 큰 건"
        for window, text in enumerate((old, "사급력이 큰 건 PHP가 인터넷 정보")):
            event = assembler.process(window, text, shared_speech=True,
                                      supported_tail_revision=True)
            partial = state.process(window, event.utterance_hypothesis, True,
                3 + 2 * window, confirmed_prefix_length=event.confirmed_prefix_length,
                source_supported=window == 0, source_text=production_source_fragment(event))[0]
        self.assertEqual(event.new_fragment, "PHP가 인터넷 정보")
        self.assertEqual(partial.text, old + " PHP가 인터넷 정보")
        self.assertEqual(partial.publication_text, old)
        self.assertEqual(state.flush(1, 5).publication_text, old)


class ProductionEvidenceRegressions(unittest.TestCase):
    def test_logged_windows_4_to_7_keep_prefix_and_defer_final(self) -> None:
        records = logged_windows()
        self.assertEqual(records[5]["vad"]["speech_duration_ms"], 2900.0)
        self.assertEqual(records[7]["vad"]["timestamps"], [[0, 37344], [43040, 48000]])
        self.assertFalse(records[7]["overlap"]["supported_tail_revision"])

        replay = LoggedWindowReplay()
        observed = {
            window: replay.process(records[window])
            for window in range(2, 8)
        }
        window_5, window_5_events = observed[5]
        self.assertEqual(window_5.match_type, "fuzzy")
        self.assertEqual(window_5.new_fragment, "맑은 날씨가 이어졌지만")
        self.assertTrue(window_5_events[0].publication_text.startswith("오전에는"))
        self.assertNotIn("이전에는", window_5.utterance_hypothesis)

        window_7, window_7_events = observed[7]
        expected = (
            "오전에는 비교적 맑은 날씨가 이어졌지만 오후부터는 일부 지역에 "
            "구름이 많아질 것으로 예상됩니다."
        )
        self.assertEqual(window_7.match_type, "sentence_head_revision")
        self.assertEqual(window_7.rollover_text, "")
        self.assertEqual(window_7.utterance_hypothesis, expected)
        self.assertEqual([event.status for event in window_7_events], ["partial"])
        self.assertEqual(replay.state.partial_text, expected)
        self.assertNotIn("많아졌습니다", replay.state.partial_text)
        self.assertEqual(
            replay.state.final_segments,
            ["오늘 서울 도심에서는 아침부터 많은 시민들이 출근길에 나섰습니다."],
        )

    def test_logged_windows_10_to_15_preserve_reply_and_fragment_revision(self) -> None:
        records = logged_windows()
        self.assertEqual(records[10]["vad"]["timestamps"][-1], [31264, 48000])
        self.assertEqual(records[11]["vad"]["timestamps"], [[32, 48000]])
        self.assertFalse(records[11]["overlap"]["supported_tail_revision"])

        replay = LoggedWindowReplay()
        observed = {}
        for window in range(8, 16):
            # The original log predates L2 and records false here. The L2 gate
            # regression derives true from these recorded VAD coordinates plus
            # synthetic shared audio; this text/state replay consumes that gate.
            observed[window] = replay.process(
                records[window],
                supported_tail_revision=True if window == 11 else None,
            )

        window_10, window_10_events = observed[10]
        self.assertEqual(window_10.match_type, "token_exact")
        self.assertEqual(window_10.utterance_hypothesis, "네, 현재까지 큰 교육을")
        self.assertEqual(window_10_events[-1].publication_text, "네, 현재까지 큰 교육을")

        window_11, _ = observed[11]
        self.assertEqual(window_11.match_type, "supported_tail_replace")
        self.assertEqual(
            window_11.utterance_hypothesis,
            "네, 현재까지 큰 교통 혼잡은 발생하지 않았습니다.",
        )
        self.assertEqual(window_11.utterance_hypothesis.count("현재까지"), 1)
        self.assertNotIn("교육을 현재까지", window_11.utterance_hypothesis)

        window_14, window_14_events = observed[14]
        corrected = (
            "한편 전문가들은 갑작스러운 기운 변화에 대비해 "
            "건강관리에 주의할 것을 당부했습니다."
        )
        self.assertEqual(window_14.match_type, "sentence_head_revision")
        self.assertEqual(window_14.rollover_text, "")
        self.assertEqual(window_14.utterance_hypothesis, corrected)
        self.assertEqual([event.status for event in window_14_events], ["partial"])

        window_15, window_15_events = observed[15]
        self.assertEqual(window_15_events[0].status, "final")
        self.assertEqual(window_15_events[0].text, corrected)
        self.assertEqual(window_15.utterance_hypothesis, "그럼요. 외출하기 전")
        self.assertEqual(window_15_events[-1].publication_text, "그럼요. 외출하기 전")

    def test_logged_windows_18_to_21_do_not_replay_finalized_text(self) -> None:
        records = logged_windows()
        self.assertEqual(records[20]["vad"]["speech_duration_ms"], 734.0)
        self.assertFalse(records[21]["vad"]["speech_detected"])

        replay = LoggedWindowReplay()
        observed = {
            window: replay.process(records[window])
            for window in (18, 19, 20, 21)
        }
        first = "오늘 준비한 소식은 여기까지입니다."
        second = "시청해 주셔서 감사합니다."

        window_19, window_19_events = observed[19]
        self.assertEqual(window_19_events[0].text, first)
        self.assertEqual(window_19.utterance_hypothesis, second)
        self.assertNotIn(first, replay.state.final_segments[-1])

        window_20, window_20_events = observed[20]
        self.assertTrue(window_20.duplicate_only)
        self.assertEqual(window_20_events[0].text, second)
        self.assertEqual(window_20.utterance_hypothesis, "")
        self.assertEqual(observed[21][1], [])
        self.assertEqual(replay.state.partial_text, "")
        self.assertEqual(replay.state.final_segments, [first, second])

    def test_supported_window_10_to_11_revision_preserves_unobserved_leading_reply(self) -> None:
        unsupported = SubtitleAssembler(0)
        unsupported.process(10, "네, 현재까지 큰 교육을")
        unsupported_event = unsupported.process(
            11,
            "현재까지 큰 교통 혼잡은 발생하지 않았습니다.",
            shared_speech=True,
            supported_tail_revision=False,
        )
        self.assertEqual(unsupported_event.match_type, "none")
        self.assertIn("큰 교육을 현재까지", unsupported_event.utterance_hypothesis)

        assembler = SubtitleAssembler(0)
        state = SpeakerSubtitleState(0, require_final_support=True)
        first = assembler.process(10, "네, 현재까지 큰 교육을")
        state.process(
            10, first.utterance_hypothesis, True, 23,
            confirmed_prefix_length=first.confirmed_prefix_length,
            source_supported=True, source_text=first.raw,
        )

        event = assembler.process(
            11,
            "현재까지 큰 교통 혼잡은 발생하지 않았습니다.",
            shared_speech=True,
            supported_tail_revision=True,
        )

        self.assertEqual(event.match_type, "supported_tail_replace")
        self.assertEqual(
            event.utterance_hypothesis,
            "네, 현재까지 큰 교통 혼잡은 발생하지 않았습니다.",
        )
        partial = state.process(
            11, event.utterance_hypothesis, True, 25,
            confirmed_prefix_length=event.confirmed_prefix_length,
            source_supported=True, source_text=production_source_fragment(event),
        )[0]
        self.assertEqual(partial.publication_text, event.utterance_hypothesis)

    def test_supported_tail_revision_preserves_independent_leading_sentence(self) -> None:
        assembler = SubtitleAssembler(0)
        assembler.process(10, "별도 안내입니다. 현재까지 큰 교육을")

        event = assembler.process(
            11,
            "현재까지 큰 교통 혼잡은 발생하지 않았습니다.",
            shared_speech=True,
            supported_tail_revision=True,
        )

        self.assertEqual(event.match_type, "supported_tail_replace")
        self.assertEqual(
            event.utterance_hypothesis,
            "별도 안내입니다. 현재까지 큰 교통 혼잡은 발생하지 않았습니다.",
        )
        self.assertEqual(event.utterance_hypothesis.count("현재까지"), 1)

    def test_supported_tail_revision_does_not_swallow_distinct_new_sentence(self) -> None:
        assembler = SubtitleAssembler(0)
        assembler.process(0, "첫 번째 안내를 마칩니다")

        event = assembler.process(
            1,
            "두 번째 소식을 전합니다",
            shared_speech=True,
            supported_tail_revision=True,
        )

        self.assertNotEqual(event.match_type, "supported_tail_replace")
        self.assertEqual(
            event.utterance_hypothesis,
            "첫 번째 안내를 마칩니다 두 번째 소식을 전합니다",
        )

    def test_supported_tail_revision_keeps_intentional_repetition(self) -> None:
        assembler = SubtitleAssembler(0)
        repeated = "안내를 마칩니다 안내를 마칩니다"
        assembler.process(0, repeated)

        event = assembler.process(
            1,
            "안내를 마칩니다",
            shared_speech=True,
            supported_tail_revision=True,
        )

        self.assertNotEqual(event.match_type, "supported_tail_replace")
        self.assertEqual(event.utterance_hypothesis, repeated)
        self.assertEqual(event.utterance_hypothesis.count("안내를 마칩니다"), 2)

    def test_unsupported_full_fuzzy_correction_preserves_prefix_and_extends_suffix(self) -> None:
        assembler = SubtitleAssembler(0)
        state = SpeakerSubtitleState(0, require_final_support=True)
        first = assembler.process(4, "오전에는 비교적")
        state.process(
            4,
            first.utterance_hypothesis,
            True,
            11,
            confirmed_prefix_length=first.confirmed_prefix_length,
            source_supported=True,
            source_text=first.raw,
        )

        event = assembler.process(
            5,
            "이전에는 비교적 맑은 날씨가 이어졌지만",
            shared_speech=True,
            supported_tail_revision=False,
        )

        self.assertEqual(event.match_type, "fuzzy")
        self.assertEqual(event.new_fragment, "맑은 날씨가 이어졌지만")
        self.assertEqual(
            event.utterance_hypothesis,
            "오전에는 비교적 맑은 날씨가 이어졌지만",
        )
        self.assertEqual(event.utterance_hypothesis.count("비교적"), 1)
        partial = state.process(
            5,
            event.utterance_hypothesis,
            True,
            13,
            confirmed_prefix_length=event.confirmed_prefix_length,
            source_supported=True,
            source_text=production_source_fragment(event),
        )[0]
        self.assertEqual(partial.publication_text, event.utterance_hypothesis)

    def test_shifted_window_head_preserves_prefix_through_publication_and_flush(self) -> None:
        # User-supplied gui_last_run examples, not an imported JSON replay.
        for old, raw, expected in (
            (
                "오늘 서울 도심에서는 아침부터",
                "그래서는 아침부터 많은 시민들이 출근길에 나섰습니다.",
                "오늘 서울 도심에서는 아침부터 많은 시민들이 출근길에 나섰습니다.",
            ),
            (
                "오전에는 비교적 말",
                "이번에는 비교적 맑은 날씨가 이어졌지만 오후부터",
                "오전에는 비교적 맑은 날씨가 이어졌지만 오후부터",
            ),
        ):
            for tail_support in (False, True):
                with self.subTest(old=old, tail_support=tail_support):
                    assembler = SubtitleAssembler(0)
                    state = SpeakerSubtitleState(0, require_final_support=True)
                    first = assembler.process(0, old)
                    state.process(0, first.utterance_hypothesis, True, 3,
                                  confirmed_prefix_length=first.confirmed_prefix_length,
                                  source_supported=True, source_text=first.raw)
                    event = assembler.process(1, raw, shared_speech=True,
                                              supported_tail_revision=tail_support)
                    self.assertEqual(event.utterance_hypothesis, expected)
                    partial = state.process(
                        1, event.utterance_hypothesis, True, 5,
                        confirmed_prefix_length=event.confirmed_prefix_length,
                        source_supported=True, source_text=production_source_fragment(event),
                    )[0]
                    self.assertEqual(partial.publication_text, expected)
                    self.assertEqual(partial.candidate_provenance, "VALIDATED")
                    self.assertEqual(state.flush(1, 5).publication_text, expected)

    def test_short_hypothesis_continuation_and_exact_extension(self) -> None:
        for old, raw, expected in (
            ("오전에는 비교적", "이번에는 비교적 맑은 날씨가 이어졌지만 오후부터",
             "이번에는 비교적 맑은 날씨가 이어졌지만 오후부터"),
            ("시민들은 대중교통을", "시민들은 대중교통을 이용하거나 도로 상황을 확인하며",
             "시민들은 대중교통을 이용하거나 도로 상황을 확인하며"),
        ):
            with self.subTest(old=old):
                assembler = SubtitleAssembler(0)
                assembler.process(0, old)
                event = assembler.process(
                    1,
                    raw,
                    shared_speech=True,
                    supported_tail_revision=False,
                )
                self.assertEqual(event.utterance_hypothesis, expected)
                if old == "오전에는 비교적":
                    self.assertEqual(event.match_type, "fuzzy_replace")

    def test_shifted_head_requires_shared_speech_and_adjacent_windows(self) -> None:
        old = "오늘 서울 도심에서는 아침부터"
        raw = "그래서는 아침부터 많은 시민들이 출근길에 나섰습니다."
        for window, shared in ((1, False), (1, None), (2, True)):
            with self.subTest(window=window, shared=shared):
                assembler = SubtitleAssembler(0)
                assembler.process(0, old)
                event = assembler.process(window, raw, shared_speech=shared)
                self.assertEqual(event.utterance_hypothesis, old + " " + raw)

    def test_disabled_shifted_head_preserves_overlap_mode_behavior(self) -> None:
        old = "오늘 서울 도심에서는 아침부터"
        raw = "그래서는 아침부터 많은 시민들이 출근길에 나섰습니다."
        assembler = SubtitleAssembler(0, allow_shifted_head=False)
        assembler.process(0, old)
        event = assembler.process(1, raw, shared_speech=True)
        self.assertEqual(event.match_type, "none")
        self.assertEqual(event.utterance_hypothesis, old + " " + raw)

    def test_shifted_head_never_removes_an_independently_confirmed_tail(self) -> None:
        old = "오전에는 비교적 말"
        raw = "이번에는 비교적 맑은 날씨가 이어졌지만 오후부터"
        assembler = SubtitleAssembler(0)
        assembler.process(0, old)
        confirmed = assembler.process(1, old, shared_speech=True)
        self.assertEqual(confirmed.confirmed_prefix_length, len(normalize_for_matching(old)))
        event = assembler.process(2, raw, shared_speech=True)
        self.assertEqual(event.utterance_hypothesis, old + " " + raw)
        self.assertEqual(event.confirmed_prefix_length, confirmed.confirmed_prefix_length)

    def test_repetition_and_unrelated_internal_words_are_preserved(self) -> None:
        for old, raw in (
            ("오늘 서울 도심에서는 아침부터", "그래서는 아침부터"),
            ("오늘 서울 도심에서는 아침부터", "새로운 아침부터 시작합니다"),
            ("오전에는 비교적 말", "이번에는 비교적 다른 내용입니다"),
            ("오전에는 비교적 말입니다", "이번에는 비교적 맑은 날씨입니다"),
            ("나는 진짜 진짜", "너는 진짜 진짜 맛있다고 말했다"),
            ("도심에서는 123456", "그래서는 123456 다음 번호입니다"),
        ):
            with self.subTest(old=old, raw=raw):
                assembler = SubtitleAssembler(0)
                assembler.process(0, old)
                event = assembler.process(1, raw, shared_speech=True)
                self.assertEqual(event.utterance_hypothesis, old + " " + raw)
        assembler = SubtitleAssembler(0)
        assembler.process(0, "오늘 서울 도심에서는 아침부터")
        event = assembler.process(1, "그래서는 아침부터 진짜 진짜 바빴습니다", shared_speech=True)
        self.assertEqual(event.utterance_hypothesis, "오늘 서울 도심에서는 아침부터 진짜 진짜 바빴습니다")

    def test_shifted_head_does_not_grant_source_support(self) -> None:
        old = "오늘 서울 도심에서는 아침부터"
        raw = "그래서는 아침부터 다음 영상에서 만나요"
        for first_supported in (False, True):
            with self.subTest(first_supported=first_supported):
                assembler = SubtitleAssembler(0)
                state = SpeakerSubtitleState(0, require_final_support=True)
                assembler.process(0, old)
                state.process(0, old, True, 3, confirmed_prefix_length=0,
                              source_supported=first_supported, source_text=old)
                event = assembler.process(1, raw, shared_speech=True)
                self.assertEqual(event.new, "다음 영상에서 만나요")
                partial = state.process(
                    1, event.utterance_hypothesis, True, 5,
                    confirmed_prefix_length=event.confirmed_prefix_length,
                    source_supported=False, source_text=production_source_fragment(event),
                )[0]
                expected = old if first_supported else ""
                self.assertEqual(partial.publication_text, expected)
                self.assertEqual(state.flush(1, 5).publication_text, expected)

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
