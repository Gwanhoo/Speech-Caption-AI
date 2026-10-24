"""Local presentation replay; no backend, capture, VAD or STT is started.

Run from the repository root: python -m phase5.replay_overlay_timeline
Wire events remain unchanged and in file order. Their millisecond timestamps
become relative arrival times on an injected presentation clock. At speed 1,
one recorded second is one wall-clock second; speed scales expiry as well.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from .gui_app import SubtitleFeedModel
from .subtitle_overlay import OverlayStyle, SubtitleOverlay
from .subtitle_presentation import FINAL_LIFETIME_SECONDS, SubtitlePresentationState
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


DEFAULT_TIMELINE = (Path(__file__).resolve().parents[1] / "phase3/output/evidence"
                    / "news_20261022_160827/timeline.jsonl")


def load_events(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events = [row["event"] for row in records if row["record_type"] == "subtitle_event"]
    if not events:
        raise ValueError("No subtitle_event records")
    stamps = [event.get("timestamp") for event in events]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in stamps):
        raise ValueError("Replay requires finite millisecond timestamps")
    if any(right < left for left, right in zip(stamps, stamps[1:])):
        raise ValueError("Timestamps must be nondecreasing in recorded file order")
    return events


class TimelineReplay:
    """Drive the production Qt model -> production overlay with virtual time."""

    def __init__(self, events: list[dict], style: OverlayStyle = OverlayStyle()) -> None:
        self.events = events
        self.offsets = [(event["timestamp"] - events[0]["timestamp"]) / 1000
                        for event in events]
        self.now = 0.0
        self.cursor = 0
        self.model = SubtitleFeedModel()
        # Deterministic replay owns time; production's independent polling timer
        # would otherwise interleave with frozen inspection snapshots.
        self.model._expiry_timer.stop()
        self.model._presentation = SubtitlePresentationState(clock=lambda: self.now)
        self.overlay = SubtitleOverlay()
        self.overlay.apply_style(style)
        self.model.entries_changed.connect(self.overlay.set_entries)
        self.accepted_sequences: list[int] = []

    def advance(self, seconds: float, *, through_sequence: int | None = None) -> None:
        if seconds < self.now:
            raise ValueError("Replay time cannot move backwards")
        while self.cursor < len(self.events) and self.offsets[self.cursor] <= seconds:
            event = self.events[self.cursor]
            if through_sequence is not None and event["sequence"] > through_sequence:
                break
            self.now = self.offsets[self.cursor]
            self.model.expire_final_entries()
            if self.model.upsert_event(event):
                self.accepted_sequences.append(event["sequence"])
            self.cursor += 1
        self.now = seconds
        self.model.expire_final_entries()

    def close(self) -> None:
        self.model.clear()
        self.overlay.close()
        self.model.deleteLater()
        self.overlay.deleteLater()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, default=DEFAULT_TIMELINE)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--font-size", type=int, default=28)
    parser.add_argument("--pause-at-sequence", type=int,
                        help="Jump through this event using recorded time, then freeze expiry")
    parser.add_argument("--close-after", type=float,
                        help="Optional wall-clock seconds until automatic window closure")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if args.close_after is not None and (not math.isfinite(args.close_after) or args.close_after <= 0):
        parser.error("--close-after must be finite and positive")
    events = load_events(args.timeline)
    if args.pause_at_sequence is not None and not any(
        event["sequence"] == args.pause_at_sequence for event in events
    ):
        parser.error("--pause-at-sequence is absent from the recording")
    app = QApplication.instance() or QApplication([])
    replay = TimelineReplay(events, OverlayStyle(font_size=args.font_size))
    replay.overlay.setWindowTitle("VOXLUX local timeline replay")
    replay.overlay.show()
    timer = QTimer()
    started = time.monotonic()
    if args.pause_at_sequence is not None:
        index = next(i for i, event in enumerate(events)
                     if event["sequence"] == args.pause_at_sequence)
        replay.advance(replay.offsets[index], through_sequence=args.pause_at_sequence)
        print(f"Frozen sequence={args.pause_at_sequence}, recorded elapsed={replay.now:.3f}s", flush=True)
    else:
        print(f"Replay {len(events)} events at {args.speed}x; origin=first subtitle timestamp. "
              "Initial pre-subtitle silence is omitted; wire timestamps are unchanged.", flush=True)
        def tick() -> None:
            replay.advance((time.monotonic() - started) * args.speed)
            if replay.now >= replay.offsets[-1] + FINAL_LIFETIME_SECONDS + 0.2:
                app.quit()
        timer.timeout.connect(tick)
        timer.start(10)
    if args.close_after is not None:
        QTimer.singleShot(round(args.close_after * 1000), app.quit)
    try:
        return app.exec()
    finally:
        timer.stop()
        replay.close()


if __name__ == "__main__":
    raise SystemExit(main())
