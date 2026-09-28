from __future__ import annotations

from subtitle_assembler import SpeakerSubtitleState, normalize_for_matching


def active(state: SpeakerSubtitleState, window: int, text: str, at: float):
    events = state.process(window, text, True, at)
    assert len(events) == 1
    return events[0]


def main() -> int:
    extension = SpeakerSubtitleState(speaker=0)
    first = active(extension, 0, "데이터베이스는 여러 사용자가", 3.0)
    second = active(
        extension,
        1,
        "데이터베이스는 여러 사용자가 필요한 정보를 저장합니다",
        5.0,
    )
    assert first.utterance_id == second.utterance_id == 1
    assert second.action == "extend"
    assert second.text == "데이터베이스는 여러 사용자가 필요한 정보를 저장합니다"
    assert second.stability_action == "promote_stable"

    punctuation = SpeakerSubtitleState(speaker=0)
    active(punctuation, 0, "안녕하세요.", 3.0)
    punctuation_extension = active(
        punctuation, 1, "안녕하세요. 오늘 날씨가 좋습니다.", 5.0
    )
    assert punctuation_extension.text == "안녕하세요. 오늘 날씨가 좋습니다."

    correction = SpeakerSubtitleState(speaker=0)
    active(correction, 0, "오늘은 학교에", 3.0)
    corrected = active(correction, 1, "오늘은 식당에", 5.0)
    assert corrected.action == "replace"
    assert corrected.text == "오늘은 식당에"
    assert corrected.stability_action == "consensus_correction"

    fragment = SpeakerSubtitleState(speaker=0)
    good = "데이터베이스는 여러 사용자가 필요한 정보를 저장합니다"
    active(fragment, 0, good, 3.0)
    retained = active(fragment, 1, good, 5.0)
    assert retained.action == "retain" and retained.text == good

    tail = SpeakerSubtitleState(speaker=0)
    active(tail, 0, "데이터베이스는 정보를 저장하고 관리하는 시스템. 입니다베스", 3.0)
    repaired = active(
        tail, 1, "데이터베이스는 정보를 저장하고 관리하는 시스템입니다", 5.0
    )
    assert repaired.action == "replace"
    assert "입니다베스" not in repaired.text
    assert normalize_for_matching(repaired.text).endswith("시스템입니다")

    silence = SpeakerSubtitleState(speaker=0, finalize_silence_ms=2000)
    active(silence, 0, "발화가 진행 중입니다", 3.0)
    finalized = silence.process(1, "", False, 5.0)
    assert len(finalized) == 1
    assert finalized[0].status == "final"
    assert finalized[0].finalize_reason == "vad_silence"

    new_utterance = active(silence, 2, "안녕하세요", 7.0)
    assert new_utterance.utterance_id == 2
    assert new_utterance.action == "start"

    repeated = SpeakerSubtitleState(speaker=0, finalize_silence_ms=2000)
    active(repeated, 0, "같은 문장을 다시 말합니다", 3.0)
    repeated.process(1, "", False, 5.0)
    repeated_again = active(repeated, 2, "같은 문장을 다시 말합니다", 7.0)
    assert repeated_again.utterance_id == 2
    assert repeated_again.text == "같은 문장을 다시 말합니다"

    short = SpeakerSubtitleState(speaker=0, finalize_silence_ms=2000)
    short_partial = active(short, 0, "네", 3.0)
    short_final = short.process(1, "", False, 5.0)[0]
    assert short_partial.text == "네" and short_final.text == "네"

    speaker_0 = SpeakerSubtitleState(speaker=0)
    speaker_1 = SpeakerSubtitleState(speaker=1)
    event_0 = active(speaker_0, 0, "첫 번째 화자입니다", 3.0)
    event_1 = active(speaker_1, 0, "두 번째 화자입니다", 3.0)
    assert event_0.speaker == 0 and event_1.speaker == 1
    assert event_0.utterance_id == event_1.utterance_id == 1
    assert speaker_0.partial_text != speaker_1.partial_text

    ending = SpeakerSubtitleState(speaker=0)
    active(ending, 0, "종료 전에 남은 자막", 3.0)
    flushed = ending.flush(0, 3.0)
    assert flushed is not None
    assert flushed.status == "final" and flushed.finalize_reason == "pipeline_end"
    assert ending.partial_text == ""

    print("subtitle state regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
