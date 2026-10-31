"""Compare assembler/state only on supplied text and synthetic evidence."""
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'phase3'), str(ROOT / 'phase4')]
import subtitle_assembler as current
from test_general_subtitle_lifecycle import GeneralSubtitleHarness
from benchmark_stt_models import score_transcript

BASE = 'b53b208fa2e5beabad097c675ff231838f808cb5'
baseline = types.ModuleType('baseline_subtitle_assembler')
sys.modules[baseline.__name__] = baseline
exec(compile(subprocess.check_output(['git', 'show', BASE + ':phase3/subtitle_assembler.py'], cwd=ROOT, text=True), BASE, 'exec'), baseline.__dict__)
CASES = [
    ('A_supplied_text_synthetic_overlap', '오늘 서울 도심에서는 날씨였습니다.', '서울 도심에서는 아침부터 많은 시민들이 출근길에', '오늘 서울 도심에서는 아침부터 많은 시민들이 출근길에', True),
    ('library_synthetic', '지금 중앙 도서관에서는 닫혔습니다.', '중앙 도서관에서는 학생들이 함께 자료를 찾고 있습니다.', '지금 중앙 도서관에서는 학생들이 함께 자료를 찾고 있습니다.', True),
    ('meeting_synthetic', '먼저 안내드립니다. 오늘 연구 모임에서는 끝났습니다.', '연구 모임에서는 새로운 실험 방법을 논의하고 있습니다.', '먼저 안내드립니다. 오늘 연구 모임에서는 새로운 실험 방법을 논의하고 있습니다.', True),
    ('B_supplied_text', '시민들은 대중교통을 이용하거나', '공교통을 이용하거나 도로 상황을 확인하며 이동하고 있습니다.', '시민들은 대중교통을 이용하거나 도로 상황을 확인하며 이동하고 있습니다.', False),
    ('C_supplied_text', '이어졌지만 오후부터는 일부 지역의 구름', '일부 지역에 구름이 많아질 것으로 예상됩니다.', '이어졌지만 오후부터는 일부 지역에 구름이 많아질 것으로 예상됩니다.', False),
    ('D_synthetic_not_recorded_RAW', '네, 현재까지 큰 교통 혼잡은 발생하지', '큰 교통 혼잡은 발생하지 않았습니다.', '네, 현재까지 큰 교통 혼잡은 발생하지 않았습니다.', False),
]


def replay(module, case):
    _, old, raw, _, _ = case
    h = GeneralSubtitleHarness()
    h.assembler = module.SubtitleAssembler(0, enable_sentence_lifecycle=True)
    h.state = module.SpeakerSubtitleState(0, require_final_support=True)
    h.speech(0, old)
    assembly, events = h.speech(1, raw, shared_speech=True, supported_tail_revision=True)
    h.silence(2)
    return {'final': ' '.join(h.state.final_segments), 'segments': h.state.final_segments,
            'match': assembly.match_type, 'second_statuses': [e.status for e in events]}


report = {'baseline': BASE, 'kind': 'text_replay_not_acoustic_accuracy', 'cases': [],
          'raw_stt_changed': False, 'live_cer_wer': None, 'live_post_capture': None}
for case in CASES:
    row = {'name': case[0], 'reference': case[3]}
    for name, module in (('before', baseline), ('after', current)):
        result = replay(module, case)
        result['scores'] = score_transcript(case[3], result['final'])
        result['incorrect_initial_final_count'] = int(case[4] and case[1] in result['segments'])
        row[name] = result
    report['cases'].append(row)
# Interleave baseline/current to reduce warmup/order bias; whole scenario includes
# two speech windows and silence finalization, with no scoring/file IO timed.
for _ in range(20):
    for module in (baseline, current):
        for case in CASES:
            replay(module, case)
timings = {'before': [], 'after': []}
for repeat in range(300):
    order = (('before', baseline), ('after', current))
    for name, module in (order if repeat % 2 else reversed(order)):
        for case in CASES:
            started = time.perf_counter()
            replay(module, case)
            timings[name].append((time.perf_counter() - started) * 1000)
report['cpu_ms_per_two_window_scenario'] = {
    name: {'n': len(values), 'mean': float(np.mean(values)), 'p50': float(np.percentile(values, 50)),
           'p95': float(np.percentile(values, 95))} for name, values in timings.items()}
Path(__file__).with_name('comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))

# Reuse actual GPU RAW/VAD evidence without loading a model or replaying audio.
probe_path = Path(__file__).with_name('gpu_probe.json')
if probe_path.is_file():
    gpu_replays = []
    for run in json.loads(probe_path.read_text())['runs']:
        pair = {}
        for name, module in (('baseline', baseline), ('current', current)):
            h = GeneralSubtitleHarness()
            h.assembler = module.SubtitleAssembler(0, enable_sentence_lifecycle=True)
            h.state = module.SpeakerSubtitleState(0, require_final_support=True)
            for row in run['rows']:
                evidence = row['overlap_evidence']
                if row['vad']['speech_detected']:
                    h.speech(row['window'], row['raw'], shared_speech=evidence.get('shared_speech'),
                             supported_tail_revision=evidence.get('supported_tail_revision', False))
                else:
                    h.silence(row['window'])
            h.state.flush(5, 13)
            pair[name] = ' '.join(h.state.final_segments)
        gpu_replays.append({'beam': run['beam'], 'repeat': run['repeat'], 'source': run['source'],
                            **pair, 'identical': pair['baseline'] == pair['current']})
    Path(__file__).with_name('gpu_replay.json').write_text(json.dumps(
        {'kind': 'real_audio_RAW_replay_no_reference', 'runs': gpu_replays},
        ensure_ascii=False, indent=2) + '\n')
