"""Read-only source evidence audit; generated diagnostics never constitute GT."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'phase3'))
from subtitle_assembler import SubtitleAssembler, SpeakerSubtitleState
from secondary_leakage_diagnostics import subtitle_overlap_evidence

OUT = Path(__file__).parent
SOURCE = Path('/workspace/voxlux_evidence_20261022')


def replay(windows):
    assembler = SubtitleAssembler(0, enable_sentence_lifecycle=True)
    state = SpeakerSubtitleState(0, require_final_support=True)
    trace = []
    for w in windows:
        i, end = w['window_id'], w['audio']['stream_end_seconds']
        logged = w['assembly_events'][0]
        overlap = logged['audio_overlap_evidence']
        a = assembler.process(i, w['admitted_stt'][0], shared_speech=overlap.get('shared_speech'),
                              supported_tail_revision=overlap.get('supported_tail_revision', False))
        events = []
        if a.rollover_text and state.partial_text:
            final = state.finalize(i, end, a.rollover_reason)
            assembler.commit_finalized_text(final.text)
            events.append(final)
        source = a.raw_fragment if a.match_type in {'fuzzy_replace', 'supported_tail_replace', 'supported_sentence_revision'} or not a.new_fragment else a.new_fragment
        # The raw VAD records lack the later admission source flag. Reuse the
        # recorded PARTIAL's flag, never infer acoustic truth from text.
        partials = [e for e in w['subtitle_state_events'] if e['status'] == 'partial']
        supported = bool(partials and partials[-1]['source_supported'])
        new = state.process(i, a.utterance_hypothesis, w['vad'][0]['speech_detected'], end,
                            confirmed_prefix_length=a.confirmed_prefix_length,
                            source_supported=supported, source_text=source)
        events.extend(new)
        for e in new:
            if e.status == 'final':
                assembler.reset_utterance(final_text=e.text)
        trace.append({'window': i, 'assembly': asdict(a), 'state_events': [asdict(e) for e in events]})
    final = state.flush(windows[-1]['window_id'], windows[-1]['audio']['stream_end_seconds'])
    return {'trace': trace, 'flush': asdict(final) if final else None, 'finals': state.final_segments}


def resources():
    connections = subprocess.check_output(['ss', '-Hnt', 'state', 'established', '( sport = :8787 )'], text=True)
    if connections.strip():
        raise RuntimeError('Production connection detected; abort diagnostic')
    used, free, util = map(int, subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(','))
    return {'used_mib': used, 'free_mib': free, 'utilization': util}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    rows = [json.loads(s) for s in (SOURCE / 'timeline.jsonl').read_text().splitlines()]
    session = rows[0]
    windows = [r for r in rows if r['record_type'] == 'window']
    audio, sr = sf.read(SOURCE / 'original.wav', dtype='float32')
    assert sr == 16000 and audio.ndim == 1 and np.isfinite(audio).all()
    assert len(audio) == session['audio']['sample_count']
    assert [w['window_id'] for w in windows] == list(range(len(windows)))
    report = {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source': str(SOURCE), 'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE/'original.wav', SOURCE/'timeline.jsonl')},
              'session': session, 'sample_rate': sr, 'samples': len(audio), 'duration': len(audio)/sr,
              'record_counts': {kind: sum(r['record_type'] == kind for r in rows) for kind in ('session', 'window', 'subtitle_event')},
              'windows': [], 'ground_truth': None, 'cer_wer': None,
              'recorded_model_configuration': None}
    previous = None
    for w in windows:
        a = w['audio']; i = w['window_id']
        start, end = a['sample_start'], a['sample_end']
        assert (start, end) == (i*32000, i*32000+48000) and end <= len(audio)
        assert (a['stream_start_seconds'], a['stream_end_seconds']) == (start/sr, end/sr)
        clip = audio[start:end]
        for v in w['vad']:
            assert all(0 <= t['start'] < t['end'] <= len(clip) for t in v['timestamps'])
        overlap = subtitle_overlap_evidence(previous, clip, windows[i-1]['vad'][0], w['vad'][0], 16000) if i else {}
        logged = w['assembly_events'][0]
        rms = float(np.sqrt(np.mean(clip*clip)))
        report['windows'].append({'window': i, 'start': start/sr, 'end': end/sr,
            'raw': w['raw_stt']['logical_speakers'][0], 'admitted': w['admitted_stt'][0],
            'wav_rms': rms, 'recorded_rms': w['vad'][0]['rms'],
            'rms_matches': bool(np.isclose(rms, w['vad'][0]['rms'], atol=1e-7)),
            'window_wav_present': (SOURCE / a['mixed_window_wav']).is_file(),
            'recorded_vad_global_seconds': [[(start+t['start'])/sr, (start+t['end'])/sr] for t in w['vad'][0]['timestamps']],
            'recorded_overlap': logged['audio_overlap_evidence'], 'recomputed_overlap': overlap,
            'recorded_confirmed_prefix': logged['confirmed_prefix_length'],
            'logged_assembly': logged, 'logged_state': w['subtitle_state_events']})
        previous = clip
    report['unprocessed_tail_seconds'] = [windows[-1]['audio']['stream_end_seconds'], len(audio)/sr]
    report['logged_publication_finals'] = [r['event'] for r in rows if r['record_type']=='subtitle_event' and r['event']['status']=='final']
    report['current_logged_raw_replay'] = replay(windows)
    for _ in range(10):
        replay(windows)
    times = []
    for _ in range(200):
        started = time.perf_counter(); replay(windows); times.append((time.perf_counter()-started)*1000)
    report['cpu_ms_per_19_window_replay_including_trace_build'] = {'n': len(times), 'mean': float(np.mean(times)), 'p50': float(np.percentile(times,50)), 'p95': float(np.percentile(times,95))}
    (OUT/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    if not args.gpu:
        return
    from faster_whisper import WhisperModel
    from stt_context import LIVE_WHISPER_OPTIONS
    from validate_end_to_end_gpu import load_silero_vad, detect_speech_activity
    before = resources()
    if before['free_mib'] < 8000 or before['utilization'] > 10:
        raise RuntimeError('GPU is not idle or lacks headroom')
    cache = ROOT/'checkpoints/faster-whisper/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo'
    revision = (cache/'refs/main').read_text().strip()
    model_path = cache/'snapshots'/revision
    assert (model_path/'model.bin').is_file()
    vad = load_silero_vad()
    model = WhisperModel(str(model_path), device='cuda', compute_type='float16', local_files_only=True)
    for word in (False, True):
        list(model.transcribe(audio[128000:176000], **LIVE_WHISPER_OPTIONS, word_timestamps=word)[0])
    gpu = {'model': str(model_path), 'faster_whisper_version': importlib.metadata.version('faster-whisper'),
           'options': LIVE_WHISPER_OPTIONS, 'compute_type': 'float16', 'before': before,
           'reference': None, 'note': 'New model estimates, not annotations of the historical RAW.', 'runs': []}
    gpu['current_full_wav_vad'] = detect_speech_activity(vad, audio, 200)
    gpu['current_window_vads'] = [detect_speech_activity(vad, audio[w['audio']['sample_start']:w['audio']['sample_end']], 200) for w in windows]
    for repeat, modes in enumerate(((False,True),(True,False))):
        for word in modes:
            for w in windows:
                if not w['vad'][0]['speech_detected']:
                    continue
                resources()
                a = w['audio']; clip = audio[a['sample_start']:a['sample_end']]
                started = time.perf_counter()
                segments, _ = model.transcribe(clip, **LIVE_WHISPER_OPTIONS, word_timestamps=word)
                segments = list(segments)
                gpu['runs'].append({'repeat':repeat,'word_timestamps':word,'window':w['window_id'],
                    'stt_seconds':time.perf_counter()-started,'raw':' '.join(s.text.strip() for s in segments),
                    'segments':[asdict(s) for s in segments]})
            (OUT/'gpu.json').write_text(json.dumps(gpu,ensure_ascii=False,indent=2)+'\n')
    resources()
    started=time.perf_counter()
    full=list(model.transcribe(audio, **LIVE_WHISPER_OPTIONS, word_timestamps=True)[0])
    gpu['full_context_diagnostic']={'seconds':time.perf_counter()-started,'segments':[asdict(s) for s in full],
                                    'note':'Different context; not GT, not production latency'}
    gpu['summary']={}
    for word in (False,True):
        ts=[r['stt_seconds'] for r in gpu['runs'] if r['word_timestamps']==word]
        gpu['summary'][str(word)]={'n':len(ts),'mean':float(np.mean(ts)),'p50':float(np.percentile(ts,50)),'p95':float(np.percentile(ts,95))}
    (OUT/'gpu.json').write_text(json.dumps(gpu,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(gpu['summary'],indent=2))


if __name__ == '__main__':
    main()
