"""Capture diagnostics without a dependency on an operating-system audio device."""

from __future__ import annotations

import numpy as np


def select_capture_channel(audio: np.ndarray, mode: str = "first") -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 2 or not value.shape[0] or not value.shape[1]:
        raise ValueError("Expected nonempty frames x channels capture")
    if not np.isfinite(value).all():
        raise ValueError("Capture contains non-finite samples")
    if mode == "first":
        mono = value[:, 0]
    elif mode == "mean":
        mono = value.mean(axis=1)
    else:
        raise ValueError(f"Unknown capture channel mode: {mode}")
    return np.ascontiguousarray(mono)


def capture_audio_diagnostics(audio: np.ndarray, mono: np.ndarray, mode: str) -> dict:
    value = np.asarray(audio, dtype=np.float64)
    mean = value.mean(axis=1)
    rms = np.sqrt(np.mean(value * value, axis=0))
    return {
        "channel_mode": mode,
        "channel_count": value.shape[1],
        "channel_rms": rms.tolist(),
        "channel_peak": np.max(np.abs(value), axis=0).tolist(),
        "channel_samples_at_or_above_one": np.sum(
            np.abs(value) >= 1.0, axis=0
        ).tolist(),
        "downmix_rms": float(np.sqrt(np.mean(mean * mean))),
        "selected_rms": float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))),
        "selected_peak": float(np.max(np.abs(mono))),
        "first_channel_silent_other_active": bool(
            rms[0] <= 1e-5 and np.max(rms) > 1e-3
        ),
        "downmix_cancellation_candidate": bool(
            np.sqrt(np.mean(mean * mean)) < np.max(rms) * 0.1 and np.max(rms) > 1e-3
        ),
    }
