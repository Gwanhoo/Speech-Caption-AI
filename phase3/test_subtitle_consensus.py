from __future__ import annotations

from subtitle_assembler import (
    CONSENSUS_HISTORY_SIZE,
    SpeakerSubtitleState,
    trim_repeated_consensus_boundary,
)


def active(state: SpeakerSubtitleState, window: int, text: str, at: float):
    events = state.process(window, text, True, at)
    assert len(events) == 1
    return events[0]


def main() -> int:
    trimmed, _ = trim_repeated_consensus_boundary("관리 하기", "기 위한 시스템")
    assert trimmed == "위한 시스템"

    progression = SpeakerSubtitleState(speaker=0, minimum_characters=3)
    active(progression, 0, "ABC DEF", 3.0)
    active(progression, 1, "ABC DEF GHI", 5.0)
    progressed = active(progression, 2, "ABC DEF GHI JKL", 7.0)
    assert progressed.text == "ABC DEF GHI JKL"
    assert progressed.stable_text == "ABC DEF GHI"
    assert progressed.tentative_text == "JKL"
    assert progressed.history_size == CONSENSUS_HISTORY_SIZE
    assert progressed.stability_action == "promote_stable"

    bad_tail = SpeakerSubtitleState(speaker=0)
    active(bad_tail, 0, "데이터베이스는 필요한 정보를", 3.0)
    active(bad_tail, 1, "데이터베이스는 필요한 정보를 저장하는 시스템 이,", 5.0)
    repaired = active(
        bad_tail, 2, "데이터베이스는 필요한 정보를 저장하는 시스템 입니다", 7.0
    )
    assert "시스템 이," not in repaired.text
    assert "시스템 입니다" in repaired.text
    assert repaired.stability_action in {"promote_stable", "consensus_correction"}

    better = SpeakerSubtitleState(speaker=0)
    active(better, 0, "데이터 베스는 여러 사용자가", 3.0)
    corrected = active(better, 1, "데이터베이스는 여러 사용자가 필요한 정보를", 5.0)
    assert corrected.action == "replace"
    assert corrected.stability_action == "consensus_correction"
    assert "데이터 베스" not in corrected.text

    fragment = SpeakerSubtitleState(speaker=0)
    good = "데이터베이스는 여러 사용자가 필요한 정보를 저장합니다"
    active(fragment, 0, good, 3.0)
    retained = active(fragment, 1, good, 5.0)
    assert retained.text == good
    assert retained.stability_action == "tentative_retained"

    short = SpeakerSubtitleState(speaker=0, finalize_silence_ms=2000)
    short_partial = active(short, 0, "네", 3.0)
    short_final = short.process(1, "", False, 5.0)[0]
    assert short_partial.text == short_final.text == "네"

    speaker_0 = SpeakerSubtitleState(speaker=0)
    speaker_1 = SpeakerSubtitleState(speaker=1)
    event_0 = active(speaker_0, 0, "첫 번째 화자", 3.0)
    event_1 = active(speaker_1, 0, "두 번째 화자", 3.0)
    active(speaker_0, 1, "첫 번째 화자의 다음 내용", 5.0)
    assert event_0.utterance_id == event_1.utterance_id == 1
    assert speaker_0.hypothesis_history != speaker_1.hypothesis_history
    assert speaker_0.tentative_text != speaker_1.tentative_text

    silence = SpeakerSubtitleState(speaker=0, finalize_silence_ms=2000)
    active(silence, 0, "발화 중입니다", 3.0)
    silence_final = silence.process(1, "", False, 5.0)
    assert len(silence_final) == 1
    assert silence_final[0].finalize_reason == "vad_silence"

    after_silence = active(silence, 2, "새로운 발화입니다", 7.0)
    assert after_silence.utterance_id == 2
    assert after_silence.history_size == 1

    ending = SpeakerSubtitleState(speaker=0)
    active(ending, 0, "종료 직전 자막", 3.0)
    ending_final = ending.flush(0, 3.0)
    assert ending_final is not None
    assert ending_final.text == "종료 직전 자막"
    assert ending_final.finalize_reason == "pipeline_end"

    repeated = SpeakerSubtitleState(speaker=0, finalize_silence_ms=2000)
    active(repeated, 0, "같은 문장을 다시 말합니다", 3.0)
    first_final = repeated.process(1, "", False, 5.0)[0]
    second_partial = active(repeated, 2, "같은 문장을 다시 말합니다", 7.0)
    assert first_final.text == second_partial.text
    assert second_partial.utterance_id == 2

    print("subtitle consensus regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
