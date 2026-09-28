from __future__ import annotations

import gc
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase0"))

from separate import CHECKPOINT_MARKER, MODEL_NAME, _load_clearvoice  # noqa: E402
from separation_recovery import infer_separator, separate_with_fp32_fallback  # noqa: E402
from stt_context import LIVE_WHISPER_OPTIONS, transcribe_base  # noqa: E402
from subtitle_assembler import SpeakerSubtitleState, SubtitleAssembler  # noqa: E402


INPUT_PATH = ROOT / "phase2" / "input" / "chunk_000_mixed_16k.wav"
WHISPER_MODEL_DIR = (
    ROOT
    / "checkpoints"
    / "faster-whisper"
    / "models--Systran--faster-whisper-base"
    / "snapshots"
    / "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
)
SAMPLE_RATE = 16_000
MEASURED_RUNS = 3
VAD_MINIMUM_SPEECH_MS = 200


def detect_speech_activity(
    model: Any, audio: np.ndarray, minimum_speech_ms: int
) -> dict[str, Any]:
    """Linux-safe equivalent of the production overlap pipeline VAD helper."""
    from silero_vad import get_speech_timestamps

    started = time.perf_counter()
    timestamps = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
        model,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=minimum_speech_ms,
    )
    speech_samples = sum(timestamp["end"] - timestamp["start"] for timestamp in timestamps)
    speech_ms = speech_samples * 1000 / SAMPLE_RATE
    return {
        "speech_detected": speech_ms >= minimum_speech_ms,
        "speech_duration_ms": speech_ms,
        "speech_ratio": speech_samples / len(audio),
        "rms": float(np.sqrt(np.mean(audio * audio))),
        "peak": float(np.max(np.abs(audio))),
        "processing_seconds": time.perf_counter() - started,
        "timestamps": timestamps,
        "audio_duration_ms": len(audio) * 1000 / SAMPLE_RATE,
    }


def query_gpu_memory() -> tuple[int | None, int | None]:
    device = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    device_used = None
    if device.returncode == 0:
        try:
            device_used = int(device.stdout.strip().splitlines()[0])
        except (IndexError, ValueError):
            pass

    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    process_used = None
    if processes.returncode == 0:
        for line in processes.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2:
                continue
            try:
                if int(fields[0]) == os.getpid():
                    process_used = int(fields[1])
                    break
            except ValueError:
                continue
    return device_used, process_used


class MemoryMonitor:
    def __init__(self) -> None:
        self.samples: list[tuple[int | None, int | None]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="nvidia-smi-monitor")

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(query_gpu_memory())
            self._stop.wait(0.05)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def peaks_since(self, index: int) -> tuple[int | None, int | None]:
        samples = self.samples[index:]
        devices = [device for device, _ in samples if device is not None]
        processes = [process for _, process in samples if process is not None]
        return max(devices) if devices else None, max(processes) if processes else None


def memory_text(memory: tuple[int | None, int | None]) -> str:
    device = "unavailable" if memory[0] is None else f"{memory[0]} MiB"
    process = "unavailable" if memory[1] is None else f"{memory[1]} MiB"
    return f"device_used={device}; process_used={process}"


def load_input() -> tuple[np.ndarray, float]:
    audio, sample_rate = sf.read(INPUT_PATH, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError(f"Expected 16 kHz mono WAV: {INPUT_PATH}")
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Invalid input WAV: {INPUT_PATH}")
    return np.ascontiguousarray(audio), len(audio) / sample_rate


def load_separator() -> tuple[Any, torch.device]:
    clear_voice = _load_clearvoice()
    separator = clear_voice(task="speech_separation", model_names=[MODEL_NAME])
    speech_model = separator.models[0]
    marker = Path(speech_model.args.checkpoint_dir) / CHECKPOINT_MARKER
    if not marker.is_file():
        raise RuntimeError(f"Pretrained checkpoint not found: {marker}")
    parameter = next(speech_model.model.parameters())
    device = parameter.device
    if speech_model.device.type != "cuda" or device.type != "cuda":
        raise RuntimeError(f"MossFormer2 is not on CUDA: {speech_model.device}, {device}")
    torch.cuda.synchronize(device)
    return separator, device


def separate_amp(separator: Any, device: torch.device, audio: np.ndarray):
    return separate_with_fp32_fallback(
        audio,
        amp_inference=lambda value: infer_separator(
            separator, value.reshape(1, -1), device, True
        ),
        fp32_inference=lambda value: infer_separator(
            separator, value.reshape(1, -1), device, False
        ),
    )


def run_pipeline(
    run: int,
    audio: np.ndarray,
    duration: float,
    separator: Any,
    device: torch.device,
    vad_model: Any,
    whisper: WhisperModel,
    monitor: MemoryMonitor,
) -> dict[str, Any]:
    memory_index = len(monitor.samples)
    total_started = time.perf_counter()
    separation = separate_amp(separator, device, audio)
    separated = tuple(
        np.ascontiguousarray(separation.output[speaker, 0, :], dtype=np.float32)
        for speaker in (0, 1)
    )

    vad_started = time.perf_counter()
    vad_results = [
        detect_speech_activity(vad_model, speaker_audio, VAD_MINIMUM_SPEECH_MS)
        for speaker_audio in separated
    ]
    vad_seconds = time.perf_counter() - vad_started

    speaker_results: list[dict[str, Any]] = []
    stt_total = 0.0
    assembly_total = 0.0
    for speaker, (speaker_audio, vad) in enumerate(zip(separated, vad_results)):
        if vad["speech_detected"]:
            stt = transcribe_base(whisper, speaker_audio)
            raw_text = stt.text
            stt_seconds = stt.elapsed_seconds
        else:
            raw_text = ""
            stt_seconds = 0.0
        stt_total += stt_seconds

        assembler = SubtitleAssembler(speaker=speaker)
        state = SpeakerSubtitleState(speaker=speaker)
        assembly_started = time.perf_counter()
        assembly = assembler.process(0, raw_text)
        partial_events = state.process(
            0,
            assembly.utterance_hypothesis,
            bool(vad["speech_detected"]),
            duration,
        )
        final_event = state.flush(0, duration)
        assembly_seconds = time.perf_counter() - assembly_started
        assembly_total += assembly_seconds
        events = [*partial_events, *([final_event] if final_event is not None else [])]

        speaker_results.append(
            {
                "speaker": speaker,
                "samples": len(speaker_audio),
                "rms": float(np.sqrt(np.mean(speaker_audio * speaker_audio))),
                "peak": float(np.max(np.abs(speaker_audio))),
                "vad": vad,
                "stt_seconds": stt_seconds,
                "raw_text": raw_text,
                "assembly_seconds": assembly_seconds,
                "assembled_text": assembly.utterance_hypothesis,
                "events": [event.to_dict() for event in events],
                "final_text": final_event.text if final_event is not None else "",
            }
        )

    total_seconds = time.perf_counter() - total_started
    peak_memory = monitor.peaks_since(memory_index)
    return {
        "run": run,
        "separation_seconds": separation.total_seconds,
        "amp_seconds": separation.amp_seconds,
        "fp32_seconds": separation.fp32_seconds,
        "used_fp32_fallback": separation.fallback_attempted,
        "vad_seconds": vad_seconds,
        "stt_seconds": stt_total,
        "assembly_seconds": assembly_total,
        "total_seconds": total_seconds,
        "rtf": total_seconds / duration,
        "peak_memory": peak_memory,
        "speakers": speaker_results,
    }


def distribution(values: list[float]) -> str:
    return (
        f"mean={statistics.mean(values):.6f}; median={statistics.median(values):.6f}; "
        f"min={min(values):.6f}; max={max(values):.6f}"
    )


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")
    if not (WHISPER_MODEL_DIR / "model.bin").is_file():
        raise FileNotFoundError(WHISPER_MODEL_DIR / "model.bin")

    audio, duration = load_input()
    before_models = query_gpu_memory()
    monitor = MemoryMonitor()
    monitor.start()
    try:
        started = time.perf_counter()
        separator, device = load_separator()
        separator_load_seconds = time.perf_counter() - started
        after_separator = query_gpu_memory()

        started = time.perf_counter()
        whisper = WhisperModel(
            str(WHISPER_MODEL_DIR), device="cuda", compute_type="float16"
        )
        whisper_load_seconds = time.perf_counter() - started
        after_whisper = query_gpu_memory()

        started = time.perf_counter()
        vad_model = load_silero_vad()
        vad_load_seconds = time.perf_counter() - started

        print("=== End-to-end GPU pipeline validation ===", flush=True)
        print(
            f"input={INPUT_PATH}; duration={duration:.6f}; sample_rate={SAMPLE_RATE}; channels=1",
            flush=True,
        )
        print(
            f"separator={MODEL_NAME}; device={device}; AMP=float16 autocast; "
            f"load_seconds={separator_load_seconds:.6f}",
            flush=True,
        )
        print(
            f"stt=base; device={whisper.model.device}; compute_type={whisper.model.compute_type}; "
            f"options={LIVE_WHISPER_OPTIONS}; load_seconds={whisper_load_seconds:.6f}",
            flush=True,
        )
        print(f"vad=Silero CPU; load_seconds={vad_load_seconds:.6f}", flush=True)
        print(f"memory_before_models: {memory_text(before_models)}", flush=True)
        print(f"memory_after_separator: {memory_text(after_separator)}", flush=True)
        print(f"memory_after_whisper: {memory_text(after_whisper)}", flush=True)

        for warmup in range(2):
            recovery = separate_amp(separator, device, audio)
            if recovery.fallback_attempted:
                raise RuntimeError("Unexpected FP32 fallback during separation warm-up")
            if warmup == 0:
                vad_warmup = detect_speech_activity(
                    vad_model,
                    np.ascontiguousarray(recovery.output[0, 0, :], dtype=np.float32),
                    VAD_MINIMUM_SPEECH_MS,
                )
                stt_warmup = transcribe_base(
                    whisper,
                    np.ascontiguousarray(recovery.output[0, 0, :], dtype=np.float32),
                )
                print(
                    f"warmup_stt_seconds={stt_warmup.elapsed_seconds:.6f}; "
                    f"warmup_vad_seconds={vad_warmup['processing_seconds']:.6f}",
                    flush=True,
                )
            print(
                f"warmup_separation_{warmup + 1}_seconds={recovery.total_seconds:.6f}",
                flush=True,
            )
        print(f"memory_after_warmup: {memory_text(query_gpu_memory())}", flush=True)

        runs = [
            run_pipeline(
                run,
                audio,
                duration,
                separator,
                device,
                vad_model,
                whisper,
                monitor,
            )
            for run in range(1, MEASURED_RUNS + 1)
        ]

        for result in runs:
            print(f"--- measured run {result['run']} ---", flush=True)
            print(
                f"separation={result['separation_seconds']:.6f}; "
                f"vad={result['vad_seconds']:.6f}; stt={result['stt_seconds']:.6f}; "
                f"assembly={result['assembly_seconds']:.6f}; "
                f"total={result['total_seconds']:.6f}; rtf={result['rtf']:.6f}; "
                f"fp32_fallback={result['used_fp32_fallback']}",
                flush=True,
            )
            print(f"inference_peak: {memory_text(result['peak_memory'])}", flush=True)
            for speaker in result["speakers"]:
                statuses = [event["status"] for event in speaker["events"]]
                print(
                    f"speaker_{speaker['speaker']}: samples={speaker['samples']}; "
                    f"rms={speaker['rms']:.6f}; peak={speaker['peak']:.6f}; "
                    f"vad_speech={speaker['vad']['speech_detected']}; "
                    f"vad_speech_ms={speaker['vad']['speech_duration_ms']:.1f}; "
                    f"vad_ratio={speaker['vad']['speech_ratio']:.6f}; "
                    f"vad_seconds={speaker['vad']['processing_seconds']:.6f}; "
                    f"stt_seconds={speaker['stt_seconds']:.6f}; "
                    f"raw_stt={speaker['raw_text']!r}; "
                    f"assembled={speaker['assembled_text']!r}; "
                    f"final={speaker['final_text']!r}; statuses={statuses}",
                    flush=True,
                )

        for field in (
            "separation_seconds",
            "vad_seconds",
            "stt_seconds",
            "assembly_seconds",
            "total_seconds",
            "rtf",
        ):
            print(f"summary_{field}: {distribution([run[field] for run in runs])}", flush=True)
        print(f"memory_processing_complete: {memory_text(query_gpu_memory())}", flush=True)

        del whisper
        del separator
        del vad_model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        print(f"memory_after_model_release: {memory_text(query_gpu_memory())}", flush=True)
    finally:
        monitor.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
