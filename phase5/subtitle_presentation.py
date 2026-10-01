"""Presentation-only state for the bounded live subtitle display.

The pipeline owns transcript assembly and utterance boundaries.  This module
only chooses which already-structured events are visible to the reader.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SubtitleFeedEntry:
    speaker_id: str
    utterance_id: int
    sequence: int
    status: str
    text: str
    timestamp: int | float | None = None


class SubtitlePresentationState:
    """Keep recent finals and one in-progress caption per logical stream.

    Event identity, rather than caption text, protects against an accidental
    duplicate delivery.  A later utterance with identical words remains valid.
    """

    MAX_FINAL_ENTRIES = 2
    MAX_ROWS = 3
    MAX_SEEN_EVENT_IDS = 128

    def __init__(self) -> None:
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
        self._final_entries.clear()
        self._partials.clear()
        self._finalized_utterances.clear()
        self._finalized_utterance_order.clear()
        self._seen_event_ids.clear()
        self._seen_event_order.clear()
        self._latest_sequence.clear()

    def apply(self, event: dict[str, Any]) -> bool:
        entry = self._entry_from_event(event)
        if entry is None:
            return False

        event_id = (
            entry.speaker_id,
            entry.utterance_id,
            entry.status,
            entry.sequence,
        )
        if event_id in self._seen_event_ids:
            return False
        if entry.sequence <= self._latest_sequence.get(entry.speaker_id, -1):
            return False

        key = (entry.speaker_id, entry.utterance_id)
        if entry.status == "partial":
            if key in self._finalized_utterances:
                return False
            current = self._partials.get(entry.speaker_id)
            if current is not None and entry.utterance_id < current.utterance_id:
                return False
            self._partials[entry.speaker_id] = entry
        else:
            if key in self._finalized_utterances:
                return False
            current = self._partials.get(entry.speaker_id)
            if current is not None and entry.utterance_id == current.utterance_id:
                del self._partials[entry.speaker_id]
            self._finalized_utterances.add(key)
            self._finalized_utterance_order.append(key)
            self._final_entries.append(entry)
            self._final_entries.sort(key=lambda value: value.sequence)
            overflow = len(self._final_entries) - self.MAX_FINAL_ENTRIES
            if overflow > 0:
                del self._final_entries[:overflow]

        self._remember_event(event_id)
        self._latest_sequence[entry.speaker_id] = entry.sequence
        return True

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
            or not text.strip()
        ):
            return None
        return SubtitleFeedEntry(
            speaker_id=speaker_id,
            utterance_id=utterance_id,
            sequence=sequence,
            status=status,
            text=text.strip(),
            timestamp=event.get("timestamp"),
        )

    @staticmethod
    def _speaker_id(value: Any) -> str | None:
        if value in (0, "speaker_0"):
            return "speaker_0"
        if value in (1, "speaker_1"):
            return "speaker_1"
        return None
