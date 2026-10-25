"""R2-B6: read-only evidence replay through the existing single GPU server.

No model loading, capture, operational configuration edits, or network outside
--server. Only complete production windows; tails are reported, never padded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'phase3'), str(ROOT / 'phase4')]
from remote_gpu_client import RemoteGPUClient  # noqa: E402
from secondary_leakage_diagnostics import (  # noqa: E402
    admit_subtitle_streams, build_window_diagnostic, resolve_subtitle_fragments,
    subtitle_overlap_evidence,
)
from speaker_mode import select_single_mode_primary  # noqa: E402
from speaker_tracking import PersistentSpeakerTracker  # noqa: E402
from stt_context import LIVE_WHISPER_OPTIONS  # noqa: E402
from subtitle_assembler import (  # noqa: E402
    SpeakerSubtitleState, SubtitleAssembler, normalize_for_matching,
)
from replay_overlap_assembler import fuzzy_boundary_candidate  # noqa: E402

SR = 16000
CONFIGS = [('baseline', 3, 2), ('A', 4, 2), ('B', 4, 3), ('C', 5, 3)]
EXPECTED_OPTIONS = {'language': 'ko', 'beam_size': 5, 'condition_on_previous_text': False}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file()}


def starts(samples, window, stride):
    if not 0 < stride < window:
        raise ValueError('Require 0 < stride < window')
    return range(0, samples - window * SR + 1, stride * SR)


def stats(values):
    return {'mean': float(np.mean(values)), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'max': float(max(values))} if values else None


def validate_health(health):
    if not (health.get('gpu_ready') and health.get('status') == 'ready'
            and health.get('processing_profile') == 'single'
            and health.get('stt_model') == 'large-v3-turbo'
            and health.get('stt_options') == EXPECTED_OPTIONS
            and LIVE_WHISPER_OPTIONS == EXPECTED_OPTIONS):
        raise RuntimeError('GPU server profile/model/options do not match the experiment')
    if health['requests']['waiting']:
        raise RuntimeError('GPU server has queued external work')


def inspect_evidence(evidence):
    audio, sr = sf.read(evidence / 'original.wav', dtype='float32')
    if sr != SR or audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError('Expected finite 16 kHz mono evidence')
    timeline = [json.loads(line) for line in (evidence / 'timeline.jsonl').read_text().splitlines()]
    windows = [row for row in timeline if row['record_type'] == 'window']
    run = json.loads((evidence / 'run.json').read_text())
    checks = []
    for row in windows:
        wav, rate = sf.read(evidence / row['audio']['mixed_window_wav'], dtype='float32')
        lo, hi = row['audio']['sample_start'], row['audio']['sample_end']
        equal = rate == sr and np.array_equal(wav, audio[lo:hi])
        if not equal:
            raise ValueError(f"Saved window {row['window_id']} differs from original")
        checks.append({'window': row['window_id'], 'start': lo / sr, 'end': hi / sr,
                       'sample_equal': equal, 'raw': row['raw_stt']['logical_speakers'],
                       'assembly': row['assembly_events'], 'state': row['subtitle_state_events']})
    expected = list(starts(len(audio), 3, 2))
    if [row['audio']['sample_start'] for row in windows] != expected:
        raise ValueError('Evidence window geometry differs from the claimed baseline')
    summary = {'sample_rate': sr, 'samples': len(audio), 'duration_seconds': len(audio) / sr,
               'subtype': sf.info(evidence / 'original.wav').subtype,
               'speaker_mode': run['speaker_mode'], 'window_checks': checks,
               'historical_model_not_recorded_at_top_level': 'stt_model' not in run,
               'historical_assembled': run['assembler']['assembled_by_speaker'],
               'historical_published_events': [r['event'] for r in timeline if r['record_type'] == 'subtitle_event'],
               'historical_performance': run['performance'],
               'geometries': [{'name': name, 'window': w, 'stride': s,
                               'count': len(starts(len(audio), w, s)),
                               'covered_end': (list(starts(len(audio), w, s))[-1] / SR + w),
                               'unprocessed_tail_seconds': len(audio) / SR - (list(starts(len(audio), w, s))[-1] / SR + w)}
                              for name, w, s in CONFIGS]}
    return audio, summary


class Replay:
    """Production single-mode ordering, using production functions unchanged."""
    def __init__(self, window, stride):
        self.overlap = (window - stride) * SR
        self.tracker = PersistentSpeakerTracker(overlap_samples=self.overlap)
        self.assemblers = [SubtitleAssembler(i, minimum_characters=6,
                           allow_shifted_head=True, enable_sentence_lifecycle=True) for i in (0, 1)]
        self.states = [SpeakerSubtitleState(i, minimum_characters=6,
                       finalize_silence_ms=2000, require_final_support=True) for i in (0, 1)]
        self.previous = None
        self.primary = None

    def process(self, index, end, mixture, raw_speakers, slots, silence):
        assignment = self.tracker.assign(window=index, raw_speakers=raw_speakers,
                                        mixture=mixture, pre_separation_silence=silence)
        speakers = assignment.speakers
        mapped = [None, None]
        for raw, slot in enumerate(slots):
            mapped[assignment.diagnostic['raw_to_logical_mapping'][str(raw)]] = slot
        texts = [slot['raw_transcript'] for slot in mapped]
        vads = [dict(slot['vad']) for slot in mapped]
        raw_texts = list(texts)
        diagnostic = build_window_diagnostic(speaker_0=speakers[0], speaker_1=speakers[1],
                                             vad_results=vads, transcripts=texts)
        primary, selection = select_single_mode_primary(assignment.diagnostic, vads)
        if primary is None:
            primary = self.primary if self.primary is not None else next(
                (v for v in assignment.diagnostic['raw_to_logical_mapping'].values() if v in (0, 1)), 0)
        else:
            self.primary = primary
        for i in (0, 1):
            if i != primary:
                texts[i] = ''
                vads[i].update(speech_detected=False, timestamps=[])
        texts, vads, routing = resolve_subtitle_fragments(
            speakers=speakers, transcripts=texts, vad_results=vads, diagnostic=diagnostic,
            previous_texts=[a.previous if a.last_window == index - 1 else '' for a in self.assemblers],
            active_hypotheses=[s.partial_text for s in self.states])
        texts, vads, admission = admit_subtitle_streams(
            mixture=mixture, speakers=speakers, transcripts=texts, vad_results=vads,
            active_hypotheses=[s.partial_text for s in self.states],
            speaker_assignment=assignment.diagnostic,
            candidate_contexts=[s.candidate_context for s in self.states], window=index)
        assembler, state = self.assemblers[primary], self.states[primary]
        events = []
        if vads[primary].get('candidate_transition', {}).get('restart'):
            discarded = state.finalize(index, end, 'secondary_candidate_restart')
            events.append(discarded.to_dict())
            assembler.reset_utterance(final_text=discarded.text)
        overlap = {}
        if self.previous is not None and self.previous[0] == index - 1:
            overlap = subtitle_overlap_evidence(self.previous[1][primary], speakers[primary],
                       self.previous[2][primary], vads[primary], self.overlap)
        event = assembler.process(index, texts[primary], shared_speech=overlap.get('shared_speech'),
                                  supported_tail_revision=overlap.get('supported_tail_revision', False))
        if event.rollover_text and state.partial_text:
            final = state.finalize(index, end, str(event.rollover_reason))
            events.append(final.to_dict())
            assembler.commit_finalized_text(final.text)
        source_text = (event.raw_fragment if event.match_type in {
            'fuzzy_replace', 'supported_tail_replace', 'supported_sentence_revision'
        } or not event.new_fragment else event.new_fragment)
        for item in state.process(index, event.utterance_hypothesis, vads[primary]['speech_detected'], end,
                confirmed_prefix_length=event.confirmed_prefix_length,
                source_supported=vads[primary].get('source_supported', False), source_text=source_text,
                candidate_transition=vads[primary].get('candidate_transition')):
            events.append(item.to_dict())
            if item.status == 'final':
                assembler.reset_utterance(final_text=item.text)
        self.previous = (index, speakers, vads)
        return {'raw': raw_texts, 'admitted': texts, 'primary': primary, 'selection': selection,
                'assignment': assignment.diagnostic, 'vad': vads,
                'raw_vad': [dict(slot['vad']) for slot in mapped], 'routing': routing,
                'admission': admission, 'overlap_evidence': overlap,
                'assembly': event.to_dict(), 'subtitle_events': events}

    def flush(self, index, end):
        result = []
        for assembler, state in zip(self.assemblers, self.states):
            final = state.flush(index, end)
            if final:
                result.append(final.to_dict())
                assembler.reset_utterance(final_text=final.text)
        return result


def suspicion_rows(rows, finals):
    """Review candidates only. Text disagreement is not an error or ground truth."""
    result = []
    for row in rows:
        a = row['assembly']
        reasons = list(a.get('review_reasons', []))
        if a['suspicious_deletion']:
            reasons.append('assembler_suspicious_deletion')
        if row['vad'][row['primary']]['speech_detected'] and not row['raw'][row['primary']]:
            reasons.append('vad_positive_empty_raw')
        if row['raw'][row['primary']] and not row['admitted'][row['primary']]:
            reasons.append('raw_rejected_by_admission')
        if a['previous'] and a['raw'] and a['match_type'] == 'none':
            candidate = fuzzy_boundary_candidate(a['previous'], a['raw'])
            if candidate:
                reasons.append('unmerged_fuzzy_boundary_candidate')
        # A prior RAW prefix can be revised away. Flag unique 6-character runs,
        # without calling them acoustic omissions.
        if row['window'] > 0 and a['previous']:
            previous = normalize_for_matching(a['previous'])
            visible = normalize_for_matching(a['session_text'])
            lost = sorted({previous[i:i + 6] for i in range(len(previous) - 5)
                           if previous[i:i + 6] not in visible})
            if lost:
                reasons.append('previous_raw_ngrams_absent_from_session')
            else:
                lost = []
        else:
            lost = []
        if reasons:
            result.append({'window': row['window'], 'start': row['start'], 'end': row['end'],
                           'reasons': reasons, 'previous': a['previous'], 'raw': a['raw'],
                           'assembled': a['session_text'], 'absent_ngrams': lost})
    for previous, current in zip(finals, finals[1:]):
        candidate = fuzzy_boundary_candidate(previous['text'], current['text'])
        if candidate:
            result.append({'window': current['window'], 'end': current['stream_time_seconds'],
                           'reasons': ['cross_final_boundary_repeat_candidate'], **candidate})
    return result


def run_one(client, audio, name, window, stride, repeat):
    replay = Replay(window, stride)
    rows = []
    virtual_finish = 0.0
    for index, start in enumerate(starts(len(audio), window, stride)):
        end = start / SR + window
        chunk = audio[start:start + window * SR]
        began = time.perf_counter()
        remote = client.process(chunk, index, stream_start_seconds=start / SR,
                                stream_end_seconds=end, latency_diagnostics=True)
        client_seconds = time.perf_counter() - began
        # Single profile must return bit-identical input and an inactive slot.
        if not np.array_equal(remote.raw_speakers[0], chunk) or np.any(remote.raw_speakers[1]):
            raise RuntimeError('Server single-profile waveform equivalence failed')
        began = time.perf_counter()
        row = replay.process(index, end, chunk, remote.raw_speakers, remote.slots,
                             remote.pre_separation_silence)
        assembly_seconds = time.perf_counter() - began
        service_seconds = client_seconds + assembly_seconds
        virtual_start = max(end, virtual_finish)
        virtual_finish = virtual_start + service_seconds
        row.update(window=index, start=start / SR, end=end,
                   audio_sha256=hashlib.sha256(chunk.tobytes()).hexdigest(),
                   stt_seconds=remote.timing['stt_seconds'], server_timing=remote.timing,
                   client_timing=remote.client_timing, client_seconds=client_seconds,
                   local_pipeline_seconds=assembly_seconds, service_seconds=service_seconds,
                   simulated_queue_wait=virtual_start - end, simulated_result_time=virtual_finish,
                   simulated_window_start_latency=virtual_finish - start / SR,
                   simulated_oldest_new_audio_latency=virtual_finish - (rows[-1]['end'] if rows else 0))
        rows.append(row)
    flush = replay.flush(rows[-1]['window'], rows[-1]['end'])
    all_events = [event for row in rows for event in row['subtitle_events']] + flush
    finals = [e for e in all_events if e['status'] == 'final' and e['publication_text']]
    visible_rows = [r for r in rows if any(e['publication_text'] for e in r['subtitle_events'])]
    summary = {'name': name, 'repeat': repeat, 'window': window, 'stride': stride,
               'window_count': len(rows), 'covered_end': rows[-1]['end'],
               'tail_seconds': len(audio) / SR - rows[-1]['end'],
               'submitted_audio_seconds': len(rows) * window,
               'stt_calls': sum(r['vad'][r['primary']]['speech_detected'] for r in rows),
               'stt_total_seconds': sum(r['stt_seconds'] for r in rows),
               'server_processing_total_seconds': sum(r['server_timing']['processing_seconds'] for r in rows),
               'stt_per_call_seconds': stats([r['stt_seconds'] for r in rows if r['stt_seconds'] > 0]),
               'service_seconds': stats([r['service_seconds'] for r in rows]),
               'server_processing_seconds': stats([r['server_timing']['processing_seconds'] for r in rows]),
               'simulated_window_start_latency': stats([r['simulated_window_start_latency'] for r in rows]),
               'simulated_oldest_new_audio_latency': stats([r['simulated_oldest_new_audio_latency'] for r in rows]),
               'first_publication_simulated_seconds': visible_rows[0]['simulated_result_time'] if visible_rows else None,
               'max_simulated_queue_wait': max(r['simulated_queue_wait'] for r in rows),
               'final_count': len(finals), 'final_text': ' '.join(e['publication_text'] for e in finals),
               'assembly_match_counts': dict(Counter(r['assembly']['match_type'] for r in rows)),
               'source_sha256': hashlib.sha256(audio.tobytes()).hexdigest()}
    return {'summary': summary, 'windows': rows, 'flush_events': flush, 'finals': finals,
            'suspicions': suspicion_rows(rows, finals)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=ROOT / 'phase3/output/evidence/news_20261022_160827')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server', default='http://127.0.0.1:8787')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    evidence, output = args.evidence.resolve(), args.output.resolve()
    if output == evidence or evidence in output.parents or output in evidence.parents:
        parser.error('Output must be separate from evidence')
    if args.repeats < 1:
        parser.error('repeats must be positive')
    output.mkdir(parents=True, exist_ok=False)
    before = hashes(evidence)
    save(output / 'source_hashes_before.json', before)
    client = None
    try:
        audio, inspection = inspect_evidence(evidence)
        save(output / 'evidence_inspection.json', inspection)
        if args.inspect_only:
            return
        client = RemoteGPUClient(args.server)
        health = client.health()
        validate_health(health)
        # Record local defaults as supporting provenance: all live requests are
        # handled by the same existing server, never by a freshly loaded model.
        from faster_whisper import WhisperModel
        metadata = {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    'health_before': health, 'options': EXPECTED_OPTIONS, 'initial_prompt': None,
                    'transcribe_signature': str(inspect.signature(WhisperModel.transcribe)),
                    'versions': {p: importlib.metadata.version(p) for p in
                                 ['faster-whisper', 'ctranslate2', 'numpy', 'soundfile', 'silero-vad']},
                    'clock': 'perf_counter; actual loopback HTTP; simulated capture timeline, serial no-drop queue',
                    'code_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [Path(__file__), ROOT / 'phase3/run_overlap_pipeline.py',
                                              ROOT / 'phase3/subtitle_assembler.py', ROOT / 'phase3/stt_context.py',
                                              ROOT / 'phase4/gpu_processing_server.py']},
                    'orders': [], 'warmup_requests': 0}
        save(output / 'metadata.json', metadata)
        # One real speech warm-up per geometry; explicitly excluded from totals.
        for _, window, _ in CONFIGS:
            client.process(audio[10 * SR:(10 + window) * SR], 0)
            metadata['warmup_requests'] += 1
        results = []
        for repeat in range(args.repeats):
            order = np.random.default_rng(610 + repeat).permutation(len(CONFIGS)).tolist()
            metadata['orders'].append([CONFIGS[i][0] for i in order])
            for i in order:
                name, window, stride = CONFIGS[i]
                validate_health(client.health())
                result = run_one(client, audio, name, window, stride, repeat)
                save(output / f'{name}_repeat_{repeat}.json', result)
                results.append(result)
                print(json.dumps(result['summary'], ensure_ascii=False), flush=True)
        metadata['health_after'] = client.health()
        validate_health(metadata['health_after'])
        expected_requests = metadata['warmup_requests'] + sum(r['summary']['window_count'] for r in results)
        metadata['expected_requests'] = expected_requests
        metadata['observed_completed_delta'] = (metadata['health_after']['requests']['completed']
                                                - health['requests']['completed'])
        if metadata['observed_completed_delta'] != expected_requests:
            raise RuntimeError('Concurrent server traffic detected; timings not isolated')
        if metadata['health_after']['requests']['failed'] != health['requests']['failed']:
            raise RuntimeError('Server failures occurred')
        if metadata['health_after']['model_load_count'] != health['model_load_count']:
            raise RuntimeError('Model was reloaded during experiment')
        save(output / 'metadata.json', metadata)
        aggregate = []
        for name, _, _ in CONFIGS:
            selected = [r for r in results if r['summary']['name'] == name]
            first = selected[0]
            stable = all([w['raw'] for w in r['windows']] == [w['raw'] for w in first['windows']]
                         and r['summary']['final_text'] == first['summary']['final_text'] for r in selected)
            aggregate.append({'name': name, 'text_identical_across_repeats': stable,
                              'runs': [r['summary'] for r in selected],
                              'stt_total_seconds': stats([r['summary']['stt_total_seconds'] for r in selected]),
                              'suspicions': first['suspicions']})
        save(output / 'summary.json', aggregate)
    except BaseException as exc:
        save(output / 'failure.json', {'type': type(exc).__name__, 'error': str(exc)})
        raise
    finally:
        if client:
            client.close()
        after = hashes(evidence)
        save(output / 'source_integrity.json', {'unchanged': before == after,
                                               'file_count': len(before), 'after': after})
        if before != after:
            raise RuntimeError('Original evidence changed during experiment')


if __name__ == '__main__':
    main()
