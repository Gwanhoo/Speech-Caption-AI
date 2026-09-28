from __future__ import annotations

from subtitle_assembler import SpeakerSubtitleState, SubtitleAssembler, normalize_for_matching


def process_window(
    assembler: SubtitleAssembler,
    state: SpeakerSubtitleState,
    window: int,
    raw: str,
    speech_detected: bool,
    stream_time_seconds: float,
):
    assembly = assembler.process(window, raw)
    events = state.process(
        window,
        assembly.utterance_hypothesis,
        speech_detected,
        stream_time_seconds,
    )
    if events and events[-1].status == "final":
        assembler.reset_utterance()
    return assembly, events


def main() -> int:
    # This mirrors the observed Live pattern: each raw window advances while
    # adjacent windows share enough text for the assembler to remove overlap.
    raw_windows = (
        "녹화를 시작하겠습니다.",
        "시작하겠습니다. 이번 영상은 변화 관련된 영상입니다.",
        "관련된 영상입니다. 최근에 영상을 올렸는데",
        "영상을 올렸는데 이번에는 자유롭게 이야기해 보겠습니다.",
        "이야기해 보겠습니다. 마지막 내용까지 전달합니다.",
    )
    assembler = SubtitleAssembler(speaker=0)
    state = SpeakerSubtitleState(speaker=0, finalize_silence_ms=2000)
    partials = []
    assemblies = []
    for window, raw in enumerate(raw_windows):
        assembly, events = process_window(
            assembler, state, window, raw, True, 3.0 + window * 2.0
        )
        assert len(events) == 1
        assemblies.append(assembly)
        partials.append(events[0])
        assert normalize_for_matching(events[0].text) == normalize_for_matching(
            assembly.assembled
        )

    assert all(
        len(normalize_for_matching(right.text)) > len(normalize_for_matching(left.text))
        for left, right in zip(partials, partials[1:])
    )
    assert sum(event.match_type != "none" for event in assemblies[1:]) == 4
    assert normalize_for_matching(partials[-1].text).count("시작하겠습니다") == 1
    assert "마지막내용까지전달합니다" in normalize_for_matching(partials[-1].text)

    final = state.flush(len(raw_windows) - 1, 13.0)
    assert final is not None
    assert "마지막내용까지전달합니다" in normalize_for_matching(final.text)

    # A VAD final resets the assembler's utterance hypothesis, so the next
    # partial never replays finalized session history.
    next_assembler = SubtitleAssembler(speaker=0)
    next_state = SpeakerSubtitleState(speaker=0, finalize_silence_ms=2000)
    _, events = process_window(
        next_assembler, next_state, 0, "첫 번째 발화입니다.", True, 3.0
    )
    assert events[0].utterance_id == 1
    _, silence_events = process_window(
        next_assembler, next_state, 1, "", False, 5.0
    )
    assert silence_events[0].finalize_reason == "vad_silence"
    _, new_events = process_window(
        next_assembler, next_state, 2, "새로운 발화입니다.", True, 7.0
    )
    assert new_events[0].utterance_id == 2
    assert normalize_for_matching(new_events[0].text) == "새로운발화입니다"

    # Speaker state and assembler history must remain independent.
    speaker_1_assembler = SubtitleAssembler(speaker=1)
    speaker_1_state = SpeakerSubtitleState(speaker=1)
    speaker_1_assembly, speaker_1_events = process_window(
        speaker_1_assembler,
        speaker_1_state,
        0,
        "두 번째 화자의 발화입니다.",
        True,
        3.0,
    )
    assert speaker_1_events[0].speaker == 1
    assert normalize_for_matching(speaker_1_assembly.assembled) not in normalize_for_matching(
        partials[-1].text
    )

    print("subtitle assembler-state integration regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
