from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


LIVE_WHISPER_OPTIONS: dict[str, Any] = {
    "language": "ko",
    "beam_size": 5,
    "condition_on_previous_text": False,
}


def validate_stt_audio(audio: np.ndarray) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 1:
        raise ValueError("STT audio must be mono")
    if not len(value):
        raise ValueError("STT audio must not be empty")
    if not np.isfinite(value).all():
        raise ValueError("STT audio contains non-finite samples")
    return np.ascontiguousarray(value)


@dataclass(frozen=True)
class SttResult:
    text: str
    elapsed_seconds: float
    segment_count: int
    average_log_probability: float | None
    maximum_no_speech_probability: float | None
    maximum_compression_ratio: float | None


def transcribe_base(
    model: Any,
    audio: np.ndarray,
    *,
    initial_prompt: str | None = None,
) -> SttResult:
    """Run the production base-model options, optionally adding a short prompt."""
    value = validate_stt_audio(audio)
    started = time.perf_counter()
    segments, _ = model.transcribe(
        value,
        **LIVE_WHISPER_OPTIONS,
        initial_prompt=initial_prompt or None,
    )
    materialized = list(segments)
    elapsed = time.perf_counter() - started
    text = " ".join(segment.text.strip() for segment in materialized).strip()
    log_probabilities = [
        float(segment.avg_logprob)
        for segment in materialized
        if np.isfinite(segment.avg_logprob)
    ]
    no_speech_probabilities = [
        float(segment.no_speech_prob)
        for segment in materialized
        if np.isfinite(segment.no_speech_prob)
    ]
    compression_ratios = [
        float(segment.compression_ratio)
        for segment in materialized
        if np.isfinite(segment.compression_ratio)
    ]
    return SttResult(
        text=text,
        elapsed_seconds=elapsed,
        segment_count=len(materialized),
        average_log_probability=(
            sum(log_probabilities) / len(log_probabilities)
            if log_probabilities
            else None
        ),
        maximum_no_speech_probability=(
            max(no_speech_probabilities) if no_speech_probabilities else None
        ),
        maximum_compression_ratio=(
            max(compression_ratios) if compression_ratios else None
        ),
    )


class SpeakerPromptContext:
    """Bounded, speaker-isolated prompt history with an inactivity reset guard."""

    def __init__(self, speaker_count: int = 2, maximum_characters: int = 80) -> None:
        if speaker_count < 1:
            raise ValueError("speaker_count must be positive")
        if maximum_characters < 1:
            raise ValueError("maximum_characters must be positive")
        self.maximum_characters = maximum_characters
        self._text = ["" for _ in range(speaker_count)]

    def _check_speaker(self, speaker: int) -> None:
        if not 0 <= speaker < len(self._text):
            raise IndexError(f"speaker index out of range: {speaker}")

    def reset(self, speaker: int) -> None:
        self._check_speaker(speaker)
        self._text[speaker] = ""

    def prompt(self, speaker: int, speech_detected: bool) -> str | None:
        self._check_speaker(speaker)
        if not speech_detected:
            self.reset(speaker)
            return None
        return self._text[speaker] or None

    def observe(self, speaker: int, text: str, speech_detected: bool) -> None:
        self._check_speaker(speaker)
        if not speech_detected:
            self.reset(speaker)
            return
        normalized = text.strip()
        if normalized:
            self._text[speaker] = normalized[-self.maximum_characters :]

