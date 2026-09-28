from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from mix import TARGET_SAMPLE_RATE, load_wav_mono_16k


MODEL_NAME = "MossFormer2_SS_16K"
CHECKPOINT_MARKER = "last_best_checkpoint"


def _load_clearvoice():
    try:
        from clearvoice import ClearVoice
    except ImportError as exc:
        raise RuntimeError(
            "clearvoice 패키지가 설치되어 있지 않습니다. "
            "README의 dependency 설치 명령을 먼저 실행해주세요."
        ) from exc
    return ClearVoice


def separate_wav(mixed_path: Path, output_dir: Path) -> tuple[Path, Path]:
    if not mixed_path.exists():
        raise FileNotFoundError(f"mixed.wav 파일을 찾을 수 없습니다: {mixed_path}")

    ClearVoice = _load_clearvoice()
    separator = ClearVoice(
        task="speech_separation",
        model_names=[MODEL_NAME],
    )
    checkpoint_dir = Path(separator.models[0].args.checkpoint_dir)
    checkpoint_marker = checkpoint_dir / CHECKPOINT_MARKER
    if not checkpoint_marker.is_file():
        raise RuntimeError(
            f"{MODEL_NAME} pretrained checkpoint를 찾을 수 없습니다: "
            f"{checkpoint_marker}. ClearVoice 다운로드가 실패했을 수 있습니다. "
            "인터넷 연결을 확인하거나 README의 안내에 따라 모델을 수동으로 다운로드해주세요."
        )

    audio = load_wav_mono_16k(mixed_path)
    batched_audio = np.reshape(audio, (1, audio.shape[0])).astype(np.float32)

    separated = separator(batched_audio, False)
    separated = np.asarray(separated)
    if separated.ndim != 3 or separated.shape[0] < 2:
        raise RuntimeError(
            f"{MODEL_NAME}의 출력 형식이 예상과 다릅니다. "
            f"예상: [speaker, batch, samples], 실제: {separated.shape}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    speaker_1_path = output_dir / "speaker_1.wav"
    speaker_2_path = output_dir / "speaker_2.wav"

    sf.write(speaker_1_path, separated[0, 0, :], TARGET_SAMPLE_RATE, subtype="PCM_16")
    sf.write(speaker_2_path, separated[1, 0, :], TARGET_SAMPLE_RATE, subtype="PCM_16")

    return speaker_1_path, speaker_2_path


def main() -> int:
    phase0_dir = Path(__file__).resolve().parent
    mixed_path = phase0_dir / "output" / "mixed.wav"
    output_dir = phase0_dir / "output"

    try:
        speaker_1_path, speaker_2_path = separate_wav(mixed_path, output_dir)
    except Exception as exc:
        print(f"Speech separation 실패: {exc}")
        return 1

    print(f"speaker_1.wav 저장 완료: {speaker_1_path}")
    print(f"speaker_2.wav 저장 완료: {speaker_2_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
