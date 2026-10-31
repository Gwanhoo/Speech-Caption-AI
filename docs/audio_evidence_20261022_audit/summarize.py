"""Summarize recorded versus regenerated observations without treating ASR as GT."""
import copy
import json
from pathlib import Path

from audit import SOURCE, replay

OUT = Path(__file__).parent
a = json.loads((OUT/'audit.json').read_text())
g = json.loads((OUT/'gpu.json').read_text())
source = [json.loads(s) for s in (SOURCE/'timeline.jsonl').read_text().splitlines()]
windows = [w for w in source if w['record_type']=='window']
runs = {r['window']:r for r in g['runs'] if r['repeat']==0 and not r['word_timestamps']}
fresh = copy.deepcopy(windows)
for w in fresh:
    raw = runs[w['window_id']]['raw'] if w['window_id'] in runs else ''
    w['admitted_stt'][0] = raw
    w['raw_stt']['logical_speakers'][0] = raw
summary = {
    'same_recorded_current_vad_timestamps': sum(v['timestamps']==w['vad'][0]['timestamps'] for v,w in zip(g['current_window_vads'],windows)),
    'same_old_new_raw_active_windows': sum(r['raw']==windows[i]['raw_stt']['logical_speakers'][0] for i,r in runs.items()),
    'new_raw_replay_note': 'Reuse recorded VAD, overlap and publication source flags; this is conditional assembler replay, not a new full remote/admission test.',
    'new_raw_replay': replay(fresh),
    'full_vad_intervals_seconds': [[t['start']/16000,t['end']/16000] for t in g['current_full_wav_vad']['timestamps']],
}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
lines = ['# 10/22 실제 창별 진단', '', 'RAW는 모델 출력이며 정답이 아니다. VAD/word 시간도 추정이다.', '',
         '| 창 | WAV 구간(s) | 기록 RAW | 현재 turbo RAW (word off, repeat 0) | 기록 match / confirmed |',
         '|---|---|---|---|---|']
for w in a['windows']:
    i=w['window']; old=w['logged_assembly']
    lines.append(f"| {i:03} | {w['start']}–{w['end']} | {w['raw'] or '(empty)'} | {runs[i]['raw'] if i in runs else '(VAD skip)'} | {old['match_type']} / {old['confirmed_prefix_length']} |")
lines += ['', '# 새 word timestamp (전체 오디오 기준 초)', '', '당시 RAW에 붙인 정렬값이 아니라 새 turbo 추론 결과다.']
for r in g['runs']:
    if r['repeat']!=0 or not r['word_timestamps']:
        continue
    lines.extend(['', f"W{r['window']:03}: " + ', '.join(f"{w['word'].strip()} [{2*r['window']+w['start']:.2f},{2*r['window']+w['end']:.2f}]" for s in r['segments'] for w in s['words'])])
(OUT/'windows.md').write_text('\n'.join(lines)+'\n')
print('VAD matches',summary['same_recorded_current_vad_timestamps'],'/19')
print('fresh conditional replay finals',summary['new_raw_replay']['finals'])
print('full VAD intervals',summary['full_vad_intervals_seconds'])
