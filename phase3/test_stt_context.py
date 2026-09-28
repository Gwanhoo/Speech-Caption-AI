from __future__ import annotations

import numpy as np

from stt_context import SpeakerPromptContext, validate_stt_audio


def expect_value_error(audio: np.ndarray) -> None:
    try:
        validate_stt_audio(audio)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def main() -> int:
    contexts = SpeakerPromptContext(speaker_count=2, maximum_characters=8)
    assert contexts.prompt(0, True) is None
    assert contexts.prompt(1, True) is None

    contexts.observe(0, "speaker0-context", True)
    contexts.observe(1, "speaker1-context", True)
    assert contexts.prompt(0, True) == "-context"
    assert contexts.prompt(1, True) == "-context"

    contexts.observe(0, "화자 영 번", True)
    contexts.observe(1, "화자 일 번", True)
    assert contexts.prompt(0, True) != contexts.prompt(1, True)

    # An inactive/silence observation suppresses the prompt and resets only
    # that speaker's utterance context.
    assert contexts.prompt(0, False) is None
    assert contexts.prompt(0, True) is None
    assert contexts.prompt(1, True) is not None

    contexts.observe(1, "", True)
    assert contexts.prompt(1, True) is not None
    contexts.observe(1, "ignored", False)
    assert contexts.prompt(1, True) is None

    valid = validate_stt_audio(np.zeros(160, dtype=np.float32))
    assert valid.dtype == np.float32 and valid.flags.c_contiguous
    expect_value_error(np.array([], dtype=np.float32))
    expect_value_error(np.array([0.0, np.nan], dtype=np.float32))
    expect_value_error(np.array([0.0, np.inf], dtype=np.float32))
    expect_value_error(np.zeros((2, 2), dtype=np.float32))

    print("STT context regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
