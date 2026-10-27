"""Presentation-only state for the bounded live subtitle display.

The pipeline owns transcript assembly and utterance boundaries.  This module
only chooses which already-structured events are visible to the reader.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Callable


FINAL_LIFETIME_SECONDS = 4.0
LOGGER = logging.getLogger("voxlux.presentation")


@dataclass(frozen=True)
class SubtitleFeedEntry:
    speaker_id: str
    utterance_id: int
    sequence: int
    status: str
    text: str
    timestamp: int | float | None = None
    session_id: str | None = None
    window_index: int | None = None


class SubtitlePresentationState:
    """Keep recent finals and one in-progress caption per logical stream.

    Event identity, rather than caption text, protects against an accidental
    duplicate delivery.  A later utterance with identical words remains valid.
    """

    MAX_FINAL_ENTRIES = 2
    MAX_ROWS = 3
    MAX_SEEN_EVENT_IDS = 128

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._final_deadlines: dict[tuple[str, int], float] = {}
        self._final_entries: list[SubtitleFeedEntry] = []
        self._partials: dict[str, SubtitleFeedEntry] = {}
        self._finalized_utterances: set[tuple[str, int]] = set()
        self._finalized_utterance_order: list[tuple[str, int]] = []
        self._seen_event_ids: set[tuple[str, int, str, int]] = set()
        self._seen_event_order: list[tuple[str, int, str, int]] = []
        self._latest_sequence: dict[str, int] = {}

    @property
    def entries(self) -> tuple[SubtitleFeedEntry, ...]:
        partials = sorted(self._partials.values(), key=lambda entry: entry.speaker_id)
        available = self.MAX_ROWS - len(partials)
        finals = sorted(self._final_entries, key=lambda entry: entry.sequence)
        return tuple(finals[-available:] + partials) if available else tuple(partials)

    def clear(self) -> None:
        self._log_decision("clear", "session_reset")
        self._final_deadlines.clear()
        self._final_entries.clear()
        self._partials.clear()
        self._finalized_utterances.clear()
        self._finalized_utterance_order.clear()
        self._seen_event_ids.clear()
        self._seen_event_order.clear()
        self._latest_sequence.clear()

    def expire(self, now: float | None = None) -> bool:
        """Expire only FINALs using local monotonic arrival time, not wire time."""
        now = self._clock() if now is None else now
        expired = {key for key, deadline in self._final_deadlines.items() if now >= deadline}
        if not expired:
            return False
        expired_entries = [entry for entry in self._final_entries
                           if (entry.speaker_id, entry.utterance_id) in expired]
        self._final_entries = [entry for entry in self._final_entries
                               if (entry.speaker_id, entry.utterance_id) not in expired]
        for key in expired:
            del self._final_deadlines[key]
        for entry in expired_entries:
            self._log_decision("expire", "final_lifetime_elapsed", entry=entry)
        # Keep sequence/finalization tombstones: expiry must not permit replay.
        return True

    def apply(self, event: dict[str, Any]) -> bool:
        entry = self._entry_from_event(event)
        if entry is None:
            self._log_decision("reject", "invalid_event", event=event)
            return False

        event_id = (
            entry.speaker_id,
            entry.utterance_id,
            entry.status,
            entry.sequence,
        )
        if event_id in self._seen_event_ids:
            self._log_decision("reject", "duplicate_event", entry=entry)
            return False
        if entry.sequence <= self._latest_sequence.get(entry.speaker_id, -1):
            self._log_decision("reject", "stale_sequence", entry=entry)
            return False

        key = (entry.speaker_id, entry.utterance_id)
        if entry.status == "partial":
            if key in self._finalized_utterances:
                self._log_decision("reject", "finalized_utterance_replay", entry=entry)
                return False
            current = self._partials.get(entry.speaker_id)
            if current is not None and entry.utterance_id < current.utterance_id:
                self._log_decision("reject", "older_utterance", entry=entry)
                return False
            self._partials[entry.speaker_id] = entry
        else:
            if key in self._finalized_utterances:
                self._log_decision("reject", "duplicate_final", entry=entry)
                return False
            current = self._partials.get(entry.speaker_id)
            if current is not None and entry.utterance_id == current.utterance_id:
                del self._partials[entry.speaker_id]
            self._finalized_utterances.add(key)
            self._finalized_utterance_order.append(key)
            if event.get("action") != "discard":
                self._final_entries.append(entry)
                self._final_deadlines[key] = self._clock() + FINAL_LIFETIME_SECONDS
            self._final_entries.sort(key=lambda value: value.sequence)
            overflow = len(self._final_entries) - self.MAX_FINAL_ENTRIES
            if overflow > 0:
                for stale in self._final_entries[:overflow]:
                    self._final_deadlines.pop((stale.speaker_id, stale.utterance_id), None)
                del self._final_entries[:overflow]

        self._remember_event(event_id)
        self._latest_sequence[entry.speaker_id] = entry.sequence
        self._log_decision("accept", "event_applied", entry=entry)
        return True

    def _log_decision(
        self,
        action: str,
        reason: str,
        *,
        entry: SubtitleFeedEntry | None = None,
        event: dict[str, Any] | None = None,
        keys: set[tuple[str, int]] | None = None,
    ) -> None:
        source: dict[str, Any] = event or {}
        LOGGER.info(
            "action=%s reason=%s speaker=%r utterance_id=%r status=%r "
            "sequence=%r text=%r keys=%r session_id=%r window_index=%r visible=%r",
            action,
            reason,
            entry.speaker_id if entry else source.get("speaker"),
            entry.utterance_id if entry else source.get("utterance_id"),
            entry.status if entry else source.get("status"),
            entry.sequence if entry else source.get("sequence"),
            entry.text if entry else source.get("text"),
            sorted(keys) if keys else None,
            entry.session_id if entry else source.get("session_id"),
            entry.window_index if entry else source.get("window_index"),
            [
                (value.speaker_id, value.utterance_id, value.status, value.sequence, value.text)
                for value in self.entries
            ],
        )

    def _remember_event(self, event_id: tuple[str, int, str, int]) -> None:
        self._seen_event_ids.add(event_id)
        self._seen_event_order.append(event_id)
        overflow = len(self._seen_event_order) - self.MAX_SEEN_EVENT_IDS
        if overflow > 0:
            for stale_event_id in self._seen_event_order[:overflow]:
                self._seen_event_ids.discard(stale_event_id)
            del self._seen_event_order[:overflow]

        finalized_overflow = len(self._finalized_utterance_order) - self.MAX_SEEN_EVENT_IDS
        if finalized_overflow > 0:
            for stale_key in self._finalized_utterance_order[:finalized_overflow]:
                self._finalized_utterances.discard(stale_key)
            del self._finalized_utterance_order[:finalized_overflow]

    @staticmethod
    def _entry_from_event(event: dict[str, Any]) -> SubtitleFeedEntry | None:
        speaker_id = SubtitlePresentationState._speaker_id(event.get("speaker"))
        utterance_id = event.get("utterance_id")
        sequence = event.get("sequence")
        status = event.get("status")
        text = event.get("text")
        if (
            speaker_id is None
            or type(utterance_id) is not int
            or type(sequence) is not int
            or sequence < 0
            or status not in {"partial", "final"}
            or not isinstance(text, str)
            or (not text.strip() and not (status == "final" and event.get("action") == "discard"))
        ):
            return None
        return SubtitleFeedEntry(
            speaker_id=speaker_id,
            utterance_id=utterance_id,
            sequence=sequence,
            status=status,
            text=text.strip(),
            timestamp=event.get("timestamp"),
            session_id=event.get("session_id"),
            window_index=event.get("window_index"),
        )

    @staticmethod
    def _speaker_id(value: Any) -> str | None:
        if value in (0, "speaker_0"):
            return "speaker_0"
        if value in (1, "speaker_1"):
            return "speaker_1"
        return None
