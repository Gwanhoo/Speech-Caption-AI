from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch


InferenceCallable = Callable[[np.ndarray], tuple[np.ndarray, float]]


# -100 dBFS RMS: catches digital silence/residue without treating quiet speech as silence.
PRE_SEPARATION_SILENCE_RMS = 1e-5


@dataclass(frozen=True)
class AudioFiniteStats:
    finite: bool
    minimum: float | None
    maximum: float | None
    rms: float | None
    peak: float | None

    def to_dict(self) -> dict[str, float | bool | None]:
        return {
            "finite": self.finite,
            "min": self.minimum,
            "max": self.maximum,
            "rms": self.rms,
            "peak": self.peak,
        }


@dataclass(frozen=True)
class SeparationRecoveryResult:
    output: np.ndarray
    input_stats: AudioFiniteStats
    amp_seconds: float
    fp32_seconds: float
    fallback_attempted: bool
    fallback_succeeded: bool
    amp_non_finite: bool

    @property
    def total_seconds(self) -> float:
        return self.amp_seconds + self.fp32_seconds


class SeparationNumericalError(RuntimeError):
    pass


class InvalidSeparationInput(SeparationNumericalError):
    def __init__(self, stats: AudioFiniteStats) -> None:
        super().__init__("Separation input contains NaN or Inf")
        self.stats = stats


class AmpNumericalError(SeparationNumericalError):
    def __init__(self, message: str, elapsed: float = 0.0) -> None:
        super().__init__(message)
        self.elapsed = elapsed


class Fp32FallbackError(SeparationNumericalError):
    def __init__(
        self,
        message: str,
        input_stats: AudioFiniteStats,
        amp_seconds: float,
        fp32_seconds: float,
    ) -> None:
        super().__init__(message)
        self.input_stats = input_stats
        self.amp_seconds = amp_seconds
        self.fp32_seconds = fp32_seconds


def finite_audio_stats(audio: np.ndarray) -> AudioFiniteStats:
    values = np.asarray(audio)
    finite = bool(values.size and np.isfinite(values).all())
    if not finite:
        finite_values = values[np.isfinite(values)].astype(np.float64, copy=False)
        if not finite_values.size:
            return AudioFiniteStats(False, None, None, None, None)
        return AudioFiniteStats(
            finite=False,
            minimum=float(np.min(finite_values)),
            maximum=float(np.max(finite_values)),
            rms=float(np.sqrt(np.mean(finite_values * finite_values))),
            peak=float(np.max(np.abs(finite_values))),
        )
    values = values.astype(np.float64, copy=False)
    return AudioFiniteStats(
        finite=True,
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        rms=float(np.sqrt(np.mean(values * values))),
        peak=float(np.max(np.abs(values))),
    )


def output_is_finite(output: np.ndarray) -> bool:
    values = np.asarray(output)
    return bool(values.size and np.isfinite(values).all())


def is_pre_separation_silence(stats: AudioFiniteStats) -> bool:
    """Return true only for finite audio at or below the conservative RMS gate."""
    return bool(stats.finite and stats.rms is not None and stats.rms <= PRE_SEPARATION_SILENCE_RMS)


def silent_separation_output(sample_count: int) -> np.ndarray:
    """Create the normal two-speaker separator shape for a silent input window."""
    return np.zeros((2, 1, sample_count), dtype=np.float32)


def infer_separator(
    separator: Any,
    batched_audio: np.ndarray,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, float]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            separated = np.asarray(separator(batched_audio, False))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if separated.ndim != 3 or separated.shape[0] < 2 or separated.shape[1] != 1:
        raise RuntimeError(f"Unexpected separation output: {separated.shape}")
    return separated, elapsed


def separate_with_fp32_fallback(
    audio: np.ndarray,
    amp_inference: InferenceCallable,
    fp32_inference: InferenceCallable,
) -> SeparationRecoveryResult:
    stats = finite_audio_stats(audio)
    if not stats.finite:
        raise InvalidSeparationInput(stats)

    try:
        amp_output, amp_seconds = amp_inference(audio)
        amp_non_finite = not output_is_finite(amp_output)
    except AmpNumericalError as exc:
        amp_output = np.empty(0, dtype=np.float32)
        amp_seconds = exc.elapsed
        amp_non_finite = True

    if not amp_non_finite:
        return SeparationRecoveryResult(
            output=amp_output,
            input_stats=stats,
            amp_seconds=amp_seconds,
            fp32_seconds=0.0,
            fallback_attempted=False,
            fallback_succeeded=False,
            amp_non_finite=False,
        )

    fp32_output, fp32_seconds = fp32_inference(audio)
    if not output_is_finite(fp32_output):
        raise Fp32FallbackError(
            "FP32 separation output contains NaN or Inf",
            input_stats=stats,
            amp_seconds=amp_seconds,
            fp32_seconds=fp32_seconds,
        )
    return SeparationRecoveryResult(
        output=fp32_output,
        input_stats=stats,
        amp_seconds=amp_seconds,
        fp32_seconds=fp32_seconds,
        fallback_attempted=True,
        fallback_succeeded=True,
        amp_non_finite=True,
    )
