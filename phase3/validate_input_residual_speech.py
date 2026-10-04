"""Recompute source evidence with real Silero; optionally run fresh MossFormer.

No capture/HTTP simulation is described as Windows validation. The default
replays a recorded repository A.wav window, NOT single_speaker_30s_16k.wav.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from silero_vad import load_silero_vad

from secondary_leakage_diagnostics import (
    admit_subtitle_streams, attach_input_residual_speech_evidence,
)
from validate_end_to_end_gpu import detect_speech_activity, load_separator, separate_amp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-separation", action="store_true")
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent
    stem = root / "test_audio/source_admission/single_a_mossformer_window0"
    metadata = json.loads(stem.with_suffix(".json").read_text())
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as fixture:
        x, p, c = [fixture[k].copy() for k in ("mixture", "owner", "candidate")]
    vad = load_silero_vad()
    def detect(audio):
        return detect_speech_activity(vad, audio, 200)
    rows = []
    def check(name, mixture, signals, *, reject_secondary, transcripts=None):
        raw_vads = [detect(np.asarray(s, dtype=np.float32)) for s in signals]
        vads = attach_input_residual_speech_evidence(mixture, signals, raw_vads, detect)
        texts = transcripts or ["실제 첫 번째 발화", "실제 두 번째 발화"]
        old_texts, _, old = admit_subtitle_streams(
            mixture=mixture, speakers=tuple(signals), transcripts=texts,
            vad_results=raw_vads, active_hypotheses=[texts[0], ""], speaker_assignment={})
        admitted, effective, decisions = admit_subtitle_streams(
            mixture=mixture, speakers=tuple(signals), transcripts=texts,
            vad_results=vads, active_hypotheses=[texts[0], ""], speaker_assignment={})
        passed = bool(admitted[0] == texts[0] and raw_vads[1]["speech_detected"]
                      and (admitted[1] == "" if reject_secondary else admitted[1] == texts[1]))
        if reject_secondary:
            passed = passed and old_texts[1] == texts[1] and decisions["streams"][1]["suppressed"]
        else:
            passed = passed and bool(decisions["streams"][1].get("candidate_independent_support"))
            passed = passed and decisions["streams"][1]["candidate_transition"] == old["streams"][1]["candidate_transition"]
            if any(row["owner_dominates"] for row in decisions["streams"][1]["source_support_checks"]):
                residual = vads[1]["input_residual_speech"]
                passed = passed and residual["available"] and residual["speech_detected"]
        rows.append(dict(case=name, passed=passed, expected_secondary_rejected=reject_secondary,
                         raw_vads=raw_vads, baseline_admission=old, admission=decisions,
                         admitted_transcripts=admitted))
        print(f"{name}: {'PASS' if passed else 'FAIL'}; "
              f"secondary={decisions['streams'][1]['reason']}", flush=True)

    check("recorded_actual_single_A_artifact", x, (p,c), reject_secondary=True,
          transcripts=metadata["raw_transcripts"])
    tracks = []
    for filename in ("A.wav", "B.wav"):
        audio, rate = sf.read(root / "test_audio" / filename, dtype="float32")
        tracks.append(resample_poly(audio, 16000, rate)[:48000])
    first, second = tracks
    for amplitude in (1., .005, .001, .00001):
        for gain in (1., -.01):
            # Real speech, deliberately ideal source estimates: this tests
            # admission/VAD sensitivity, not MossFormer's quiet-source recall.
            check(f"real_speech_ideal_sources_{amplitude:g}_gain_{gain:g}",
                  gain*(first + amplitude*second), (gain*first, -second), reject_secondary=False)
    check("real_weak_speech_with_owner_crosstalk", first+.001*second,
          (first+.01*second, second), reject_secondary=False)
    if args.fresh_separation:
        separator, device = load_separator()
        from faster_whisper import WhisperModel
        model_root = root.parent / "checkpoints/faster-whisper/models--Systran--faster-whisper-small/snapshots"
        snapshot = next(p for p in sorted(model_root.iterdir()) if (p / "model.bin").is_file())
        whisper = WhisperModel(str(snapshot), device="cuda", compute_type="float16")
        from stt_context import transcribe_base
        for name, mixture, reject in (("fresh_single_A", first, True),
                                      ("fresh_two_speakers", .5*(first+second), False)):
            separated = separate_amp(separator, device, mixture)
            signals = [separated.output[k, 0, :] for k in (0,1)]
            if reject and abs(np.corrcoef(mixture, signals[0])[0,1]) < abs(np.corrcoef(mixture, signals[1])[0,1]):
                signals.reverse()
            transcripts = [transcribe_base(whisper, s).text for s in signals]
            check(name, mixture, signals, reject_secondary=reject, transcripts=transcripts)
    report = {"scope": "Linux real models and repository A/B speech; NOT Windows w12/WASAPI",
              "fresh_separation": args.fresh_separation, "cases": rows,
              "passed": all(row["passed"] for row in rows)}
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
