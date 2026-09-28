from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from subtitle_assembler import (
    DEFAULT_FINALIZE_SILENCE_MS,
    SpeakerSubtitleState,
    SubtitleAssembler,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "phase3" / "output" / "two_speaker_subtitle_state_30s.json"
DEFAULT_OUTPUT = ROOT / "phase3" / "output" / "subtitle_consensus_replay.json"
KNOWN_FRAGMENTS = (
    "입니다 데이터베스",
    "입니다베스",
    "에이터베스",
    "가 겠오",
    "올가 오",
    "이어 뜨 겠",
)


def count_fragments(texts: list[str]) -> dict[str, int]:
    combined = "\n".join(texts)
    return {fragment: combined.count(fragment) for fragment in KNOWN_FRAGMENTS}


def replay(raw: dict[str, Any], finalize_silence_ms: int) -> dict[str, Any]:
    states = [
        SpeakerSubtitleState(speaker=speaker, finalize_silence_ms=finalize_silence_ms)
        for speaker in (0, 1)
    ]
    assemblers = [SubtitleAssembler(speaker=speaker) for speaker in (0, 1)]
    events: list[dict[str, Any]] = []
    windows = sorted(raw["windows"], key=lambda item: item["window"])
    for window in windows:
        for speaker, transcript in enumerate(window["transcripts"]):
            assembly = assemblers[speaker].process(window["window"], transcript)
            vad_rows = window.get("vad", [])
            speech_detected = (
                bool(vad_rows[speaker]["speech_detected"])
                if speaker < len(vad_rows)
                else bool(transcript)
            )
            state_events = [
                event.to_dict()
                for event in states[speaker].process(
                    window=window["window"],
                    hypothesis=assembly.utterance_hypothesis,
                    speech_detected=speech_detected,
                    stream_time_seconds=window["stream_end_seconds"],
                )
            ]
            events.extend(state_events)
            if state_events and state_events[-1]["status"] == "final":
                assemblers[speaker].reset_utterance()

    if windows:
        for state in states:
            event = state.flush(windows[-1]["window"], windows[-1]["stream_end_seconds"])
            if event is not None:
                events.append(event.to_dict())

    old_segments = raw.get("subtitle_stabilization", {}).get(
        "final_segments_by_speaker", [[], []]
    )
    old_final = [" ".join(segments) for segments in old_segments]
    new_segments = [state.final_segments for state in states]
    new_final = [" ".join(segments) for segments in new_segments]
    return {
        "phase": "4-I-offline-replay",
        "source": str(DEFAULT_INPUT),
        "source_phase": raw.get("phase"),
        "window_count": len(windows),
        "event_count": len(events),
        "status_counts": dict(Counter(event["status"] for event in events)),
        "action_counts": dict(Counter(event["action"] for event in events)),
        "stability_action_counts": dict(
            Counter(event["stability_action"] for event in events)
        ),
        "finalize_reason_counts": dict(
            Counter(
                event["finalize_reason"]
                for event in events
                if event["finalize_reason"] is not None
            )
        ),
        "phase_4h_final_segments_by_speaker": old_segments,
        "phase_4i_final_segments_by_speaker": new_segments,
        "phase_4h_known_fragment_occurrences": count_fragments(old_final),
        "phase_4i_known_fragment_occurrences": count_fragments(new_final),
        "events": events,
        "note": (
            "This replay compares state flow and specified fragment accumulation only. "
            "No WER or semantic accuracy is inferred without ground truth."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Phase 4-I consensus replay")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--subtitle-finalize-silence-ms",
        type=int,
        default=DEFAULT_FINALIZE_SILENCE_MS,
    )
    args = parser.parse_args()
    if args.subtitle_finalize_silence_ms < 1:
        parser.error("--subtitle-finalize-silence-ms must be positive")
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    report = replay(raw, args.subtitle_finalize_silence_ms)
    report["source"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "window_count",
                    "event_count",
                    "status_counts",
                    "action_counts",
                    "stability_action_counts",
                    "finalize_reason_counts",
                    "phase_4h_known_fragment_occurrences",
                    "phase_4i_known_fragment_occurrences",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
