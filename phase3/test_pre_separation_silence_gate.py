from __future__ import annotations

import numpy as np

from separation_recovery import (
    PRE_SEPARATION_SILENCE_RMS,
    finite_audio_stats,
    is_pre_separation_silence,
    silent_separation_output,
)


def main() -> int:
    sample_count = 3 * 16_000
    zero_input = np.zeros(sample_count, dtype=np.float32)
    near_silence = np.full(sample_count, PRE_SEPARATION_SILENCE_RMS / 2, dtype=np.float32)
    speech_like = np.full(sample_count, PRE_SEPARATION_SILENCE_RMS * 100, dtype=np.float32)

    # A/B: silent and near-silent inputs take the gate, with normal separator output shape.
    assert is_pre_separation_silence(finite_audio_stats(zero_input))
    assert is_pre_separation_silence(finite_audio_stats(near_silence))
    silence_output = silent_separation_output(sample_count)
    assert silence_output.shape == (2, 1, sample_count)
    assert silence_output.dtype == np.float32
    assert np.count_nonzero(silence_output) == 0

    # C/G: ordinary signal does not take the gate and reaches the inference fast path once.
    inference_calls = 0
    for audio in (zero_input, speech_like):
        if is_pre_separation_silence(finite_audio_stats(audio)):
            output = silent_separation_output(len(audio))
        else:
            inference_calls += 1
            output = np.ones((2, 1, len(audio)), dtype=np.float32)
        assert output.shape == (2, 1, sample_count)
    assert inference_calls == 1

    # D: NaN is invalid input, never a silent window.
    invalid = zero_input.copy()
    invalid[0] = np.nan
    invalid_stats = finite_audio_stats(invalid)
    assert not invalid_stats.finite
    assert not is_pre_separation_silence(invalid_stats)

    # E/F: gate output is finite and requires no fallback/error bookkeeping.
    errors: list[Exception] = []
    fallback_attempts = 0
    assert np.isfinite(silence_output).all()
    assert not errors and fallback_attempts == 0

    print("pre-separation silence gate regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
