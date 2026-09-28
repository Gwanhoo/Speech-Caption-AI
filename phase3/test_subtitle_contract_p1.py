from __future__ import annotations

from subtitle_assembler import SpeakerSubtitleState, SubtitleAssembler, normalize_for_matching


def observe(
    assembler: SubtitleAssembler,
    state: SpeakerSubtitleState,
    window: int,
    raw_fragment: str,
    speech_detected: bool,
    at: float,
):
    assembly = assembler.process(window, raw_fragment)
    events = state.process(
        window,
        assembly.utterance_hypothesis,
        speech_detected,
        at,
    )
    if events and events[-1].status == "final":
        assembler.reset_utterance()
    return assembly, events


def main() -> int:
    # TEST 1 — simple incremental extension without boundary duplicates.
    assembler = SubtitleAssembler(speaker=0)
    state = SpeakerSubtitleState(speaker=0)
    simple = (
        "오늘은 아침에",
        "아침에 일어나서",
        "일어나서 학교에 갔습니다",
    )
    events = []
    for window, raw in enumerate(simple):
        assembly, state_events = observe(
            assembler, state, window, raw, True, 3.0 + window * 2.0
        )
        assert assembly.raw_fragment == raw
        assert len(state_events) == 1
        events.append(state_events[0])
    final_simple = normalize_for_matching(events[-1].text)
    assert final_simple == "오늘은아침에일어나서학교에갔습니다"
    assert final_simple.count("아침에") == 1
    assert final_simple.count("일어나서") == 1
    assert [event.action for event in events] == ["start", "extend", "extend"]

    # TEST 2 — a conservative fuzzy suffix/prefix boundary.
    fuzzy = SubtitleAssembler(speaker=0)
    fuzzy.process(0, "집에서 토마토 파스타를 만들어봤는데")
    fuzzy_event = fuzzy.process(
        1, "봤는데 처음 시도한 거치고는 생각보다 맛있었습니다"
    )
    fuzzy_text = normalize_for_matching(fuzzy_event.utterance_hypothesis)
    assert fuzzy_event.match_type in {"boundary_exact", "fuzzy"}
    assert fuzzy_text.count("봤는데") == 1
    assert fuzzy_text.endswith("처음시도한거치고는생각보다맛있었습니다")

    # TEST 3 — repetition inside one RAW fragment is speech, not a boundary duplicate.
    repeated = SubtitleAssembler(speaker=0).process(0, "진짜 진짜 맛있다")
    assert repeated.utterance_hypothesis == "진짜 진짜 맛있다"
    assert normalize_for_matching(repeated.utterance_hypothesis) == "진짜진짜맛있다"

    # TEST 4 — VAD final resets only the current utterance, not session history.
    reset_assembler = SubtitleAssembler(speaker=0)
    reset_state = SpeakerSubtitleState(speaker=0, finalize_silence_ms=2000)
    _, first_events = observe(
        reset_assembler, reset_state, 0, "안녕하세요.", True, 3.0
    )
    _, final_events = observe(reset_assembler, reset_state, 1, "", False, 5.0)
    second_assembly, second_events = observe(
        reset_assembler, reset_state, 2, "오늘 날씨가 좋습니다.", True, 7.0
    )
    assert first_events[0].utterance_id == 1
    assert final_events[0].status == "final"
    assert second_events[0].utterance_id == 2
    assert normalize_for_matching(second_events[0].text) == "오늘날씨가좋습니다"
    assert "안녕하세요" not in second_events[0].text
    assert "안녕하세요" in second_assembly.session_text

    # TEST 5 — each speaker owns an independent assembler and subtitle state.
    assemblers = [SubtitleAssembler(0), SubtitleAssembler(1)]
    states = [SpeakerSubtitleState(0), SpeakerSubtitleState(1)]
    _, speaker_0_events = observe(
        assemblers[0], states[0], 0, "오늘은 아침에 산책합니다", True, 3.0
    )
    _, speaker_1_events = observe(
        assemblers[1], states[1], 0, "저는 주말마다 요리합니다", True, 3.0
    )
    assert "주말마다" not in speaker_0_events[0].text
    assert "아침에" not in speaker_1_events[0].text

    # TEST 6 — a changed current-utterance hypothesis is a correction.
    correction = SpeakerSubtitleState(speaker=0)
    correction.process(0, "오늘은 학교에", True, 3.0)
    corrected = correction.process(1, "오늘은 식당에", True, 5.0)[0]
    assert corrected.action == "replace"
    assert corrected.stability_action == "consensus_correction"
    assert corrected.text == "오늘은 식당에"

    # TEST 7 — finalized session history never re-enters a new PARTIAL hypothesis.
    assert normalize_for_matching(second_assembly.session_text).startswith("안녕하세요")
    assert normalize_for_matching(second_assembly.utterance_hypothesis) == "오늘날씨가좋습니다"
    assert normalize_for_matching(second_events[0].raw_text) == "오늘날씨가좋습니다"

    print("P1 subtitle contract regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
