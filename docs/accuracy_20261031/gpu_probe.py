"""Bounded, offline-only beam probe; no reference/accuracy claims or server calls."""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'phase3'), str(ROOT / 'phase4')]
from stt_context import LIVE_WHISPER_OPTIONS
from secondary_leakage_diagnostics import subtitle_overlap_evidence
from validate_end_to_end_gpu import detect_speech_activity, load_silero_vad
from test_general_subtitle_lifecycle import GeneralSubtitleHarness

OUT = Path(__file__).with_name('gpu_probe.json')


def resource_check():
    connections = subprocess.check_output(
        ['ss', '-Hnt', 'state', 'established', '( sport = :8787 )'], text=True)
    if connections.strip():
        raise RuntimeError('Production server has established connections; stop probe')
    util, used, free = map(int, subprocess.check_output(
        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free',
         '--format=csv,noheader,nounits'], text=True).strip().split(','))
    return {'utilization': util, 'used_mib': used, 'free_mib': free}


def process_memory():
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
         '--format=csv,noheader,nounits'], text=True).splitlines()
    return next((int(r.split(',')[1]) for r in rows
                 if int(r.split(',')[0]) == os.getpid()), None)


def main():
    before = resource_check()
    if before['utilization'] > 10 or before['free_mib'] < 8000:
        raise RuntimeError('GPU is busy or lacks headroom')
    sources = []
    vad_model = load_silero_vad()
    for name in ('A', 'B'):
        path = ROOT / 'phase3/test_audio' / (name + '.wav')
        audio, sr = sf.read(path, dtype='float32')
        # Existing fixtures are mono 24kHz; exact rational 2/3 resampling.
        assert sr == 24000 and audio.ndim == 1
        audio = resample_poly(audio, 2, 3).astype(np.float32)
        windows = [np.ascontiguousarray(audio[i * 32000:i * 32000 + 48000]) for i in range(6)]
        sources.append((name, windows, [detect_speech_activity(vad_model, w, 200) for w in windows]))
    path = next((ROOT / 'checkpoints/faster-whisper').glob('models--*/snapshots/*/model.bin')).parent
    model = WhisperModel(str(path), device='cuda', compute_type='float16', local_files_only=True)
    for beam in (5, 1, 3):
        list(model.transcribe(sources[0][1][0], **{**LIVE_WHISPER_OPTIONS, 'beam_size': beam})[0])
    report = {
        'kind': 'bounded_existing_audio_gpu_probe_not_live_validation',
        'model_path': str(path.relative_to(ROOT)), 'model': 'large-v3-turbo',
        'options': LIVE_WHISPER_OPTIONS, 'compute_type': 'float16',
        'before': before, 'inputs': [], 'runs': [],
        'reference': None, 'raw_cer_wer': None, 'final_cer_wer': None,
        'hallucinations': None, 'boundary_errors': None,
        'note': 'First six complete 3s/2s windows from each existing WAV; human reference unavailable. VAD plus assembler/state replay, no capture/network/GUI.',
    }
    for name, _, _ in sources:
        p = ROOT / 'phase3/test_audio' / (name + '.wav')
        report['inputs'].append({'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    for repeat, order in enumerate(((5, 1, 3), (3, 1, 5))):
        for beam in order:
            resource_check()
            for name, windows, vads in sources:
                harness = GeneralSubtitleHarness()
                rows = []
                for i, (audio, vad) in enumerate(zip(windows, vads)):
                    resource_check()
                    started = time.perf_counter()
                    segments = list(model.transcribe(audio, **{**LIVE_WHISPER_OPTIONS, 'beam_size': beam})[0]) if vad['speech_detected'] else []
                    seconds = time.perf_counter() - started
                    raw = ' '.join(s.text.strip() for s in segments).strip()
                    evidence = subtitle_overlap_evidence(windows[i-1], audio, vads[i-1], vad, 16000) if i else {}
                    if vad['speech_detected']:
                        assembly, events = harness.speech(i, raw, shared_speech=evidence.get('shared_speech'), supported_tail_revision=evidence.get('supported_tail_revision', False))
                    else:
                        assembly, events = harness.silence(i)
                    rows.append({'window': i, 'raw': raw, 'stt_seconds': seconds, 'vad': vad,
                                 'assembly': assembly.to_dict(), 'events': [e.to_dict() for e in events],
                                 'overlap_evidence': evidence, 'process_vram_mib': process_memory()})
                final = harness.state.flush(5, 13)
                report['runs'].append({'repeat': repeat, 'beam': beam, 'source': name, 'rows': rows,
                                      'final_transcript': ' '.join(harness.state.final_segments),
                                      'flush': final.to_dict() if final else None})
            OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    report['summary'] = {}
    for beam in (5, 1, 3):
        rows = [r for run in report['runs'] if run['beam'] == beam for r in run['rows'] if r['vad']['speech_detected']]
        times = [r['stt_seconds'] for r in rows]
        report['summary'][beam] = {'inferences': len(times), 'mean': float(np.mean(times)),
                                 'p50': float(np.percentile(times, 50)), 'p95': float(np.percentile(times, 95)),
                                 'max_sampled_process_vram_mib': max(r['process_vram_mib'] or 0 for r in rows)}
    del model
    gc.collect()
    report['after'] = resource_check()
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
