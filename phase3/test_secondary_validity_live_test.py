from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from run_secondary_validity_live_test import (
    CAPTURE_SECONDS,
    SCHEDULE,
    ScheduleLog,
    inspect_wav,
    load_for_playback,
    pipeline_command,
    required_source_seconds,
    validate_schedule,
)


def main() -> None:
    validate_schedule()
    assert SCHEDULE[0].start_seconds == 0.0
    assert SCHEDULE[-1].end_seconds == CAPTURE_SECONDS == 75.0
    assert required_source_seconds() == {"A": 20.0, "B": 20.0}

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        wav = root / "source.wav"
        source_rate = 24_000
        samples = np.linspace(-0.2, 0.2, source_rate * 21, dtype=np.float32)
        sf.write(wav, samples, source_rate, subtype="PCM_16")
        info = inspect_wav(wav, minimum_seconds=20.0)
        assert info.sample_rate == source_rate
        assert info.channels == 1
        assert info.duration_seconds >= 20.0
        playback = load_for_playback(wav, duration_seconds=20.0)
        assert playback.shape == (960_000, 2)
        assert playback.dtype == np.float32
        assert np.allclose(playback[:, 0], playback[:, 1])

        short = root / "short.wav"
        sf.write(short, samples[: source_rate * 19], source_rate, subtype="PCM_16")
        try:
            inspect_wav(short, minimum_seconds=20.0)
        except ValueError as exc:
            assert "needs at least" in str(exc)
        else:
            raise AssertionError("short source was accepted")

        schedule_path = root / "schedule.json"
        log = ScheduleLog(schedule_path, SCHEDULE)
        log.set_metadata(test=True)
        log.event("CAPTURE_READY", test_start=None)
        saved = json.loads(schedule_path.read_text(encoding="utf-8"))
        assert saved["capture_duration_seconds"] == 75.0
        assert saved["schedule"][1]["kind"] == "A_only"
        assert saved["events"][0]["event"] == "CAPTURE_READY"

    command = pipeline_command(Path("C:/temporary/result.json"))
    assert "--live" in command and "--diagnostic-audio" in command
    assert command[command.index("--duration") + 1] == "75"
    assert command[command.index("--stt-backend") + 1] == "whisper"
    print("secondary validity live helper tests: PASS")


if __name__ == "__main__":
    main()
