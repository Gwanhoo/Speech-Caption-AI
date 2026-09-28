from __future__ import annotations

import numpy as np

from separation_recovery import (
    Fp32FallbackError,
    InvalidSeparationInput,
    separate_with_fp32_fallback,
)


def inference(output: np.ndarray, elapsed: float):
    def run(_: np.ndarray) -> tuple[np.ndarray, float]:
        return output.copy(), elapsed

    return run


def main() -> int:
    audio = np.linspace(-0.5, 0.5, 16, dtype=np.float32)
    finite = np.ones((2, 1, 16), dtype=np.float32)
    nan_output = finite.copy()
    nan_output[0, 0, 0] = np.nan
    inf_output = finite.copy()
    inf_output[1, 0, 1] = np.inf

    fp32_calls = 0

    def unused_fp32(_: np.ndarray) -> tuple[np.ndarray, float]:
        nonlocal fp32_calls
        fp32_calls += 1
        return finite.copy(), 0.2

    fast = separate_with_fp32_fallback(audio, inference(finite, 0.1), unused_fp32)
    assert not fast.fallback_attempted and fp32_calls == 0
    assert fast.total_seconds == 0.1

    recovered_nan = separate_with_fp32_fallback(
        audio, inference(nan_output, 0.1), inference(finite, 0.2)
    )
    assert recovered_nan.fallback_attempted and recovered_nan.fallback_succeeded
    assert recovered_nan.amp_non_finite
    assert np.isclose(recovered_nan.total_seconds, 0.3)

    recovered_inf = separate_with_fp32_fallback(
        audio, inference(inf_output, 0.1), inference(finite, 0.25)
    )
    assert recovered_inf.fallback_succeeded

    failures: list[Exception] = []
    try:
        separate_with_fp32_fallback(
            audio, inference(nan_output, 0.1), inference(nan_output, 0.2)
        )
    except Fp32FallbackError as exc:
        failures.append(exc)
    assert len(failures) == 1

    next_window = separate_with_fp32_fallback(
        audio, inference(finite, 0.1), inference(finite, 0.2)
    )
    assert not next_window.fallback_attempted

    invalid_audio = audio.copy()
    invalid_audio[3] = np.nan
    try:
        separate_with_fp32_fallback(
            invalid_audio, inference(finite, 0.1), inference(finite, 0.2)
        )
    except InvalidSeparationInput as exc:
        assert not exc.stats.finite
    else:
        raise AssertionError("Invalid separation input was not rejected")

    fatal_errors: list[Exception] = []
    successful_recovery = separate_with_fp32_fallback(
        audio, inference(nan_output, 0.1), inference(finite, 0.2)
    )
    if not successful_recovery.fallback_succeeded:
        fatal_errors.append(RuntimeError("unexpected recovery failure"))
    assert not fatal_errors

    print("separation recovery regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
