"""Export review tables from completed R2-B6 runs; never scores against a transcript."""
import csv
import json
from pathlib import Path

import numpy as np

from compare_windows import CONFIGS, save

BASE = Path(__file__).parent
RESULTS = BASE / 'results'


def read(name):
    return json.loads((RESULTS / name).read_text())


def main():
    integrity = read('source_integrity.json')
    assert integrity['unchanged']
    assert not (RESULTS / 'failure.json').exists()
    meta = read('metadata.json')
    assert meta['expected_requests'] == meta['observed_completed_delta'] == 193
    aggregate = read('summary.json')
    assert len(aggregate) == 4
    metrics, window_csv = [], []
    detail = ['# R2-B6 RAW·조립·FINAL 원문', '',
              '최종 측정 repeat 0. 반복 간 동일성은 metrics.json 참조. 모든 텍스트는 모델/조립 출력이며 정답 전사본이 아니다.', '']
    for name, window, stride in CONFIGS:
        runs = [read(f'{name}_repeat_{i}.json') for i in range(3)]
        first = runs[0]
        pooled = [w for r in runs for w in r['windows']]
        service = [w['service_seconds'] for w in pooled]
        m = {'name': name, 'window': window, 'stride': stride,
             'windows_per_run': first['summary']['window_count'],
             'stt_calls_per_run': first['summary']['stt_calls'],
             'covered_end': first['summary']['covered_end'], 'tail_seconds': first['summary']['tail_seconds'],
             'submitted_audio_seconds': first['summary']['submitted_audio_seconds'],
             'stt_input_audio_seconds': first['summary']['stt_calls'] * window,
             'stt_total_mean_s': np.mean([r['summary']['stt_total_seconds'] for r in runs]).item(),
             'stt_total_min_s': min(r['summary']['stt_total_seconds'] for r in runs),
             'stt_total_max_s': max(r['summary']['stt_total_seconds'] for r in runs),
             'server_total_mean_s': np.mean([r['summary']['server_processing_total_seconds'] for r in runs]).item(),
             'service_mean_s': np.mean(service).item(), 'service_p95_s': np.percentile(service, 95).item(),
             'window_start_latency_mean_s': np.mean([w['simulated_window_start_latency'] for w in pooled]).item(),
             'oldest_new_audio_steady_mean_s': np.mean([w['simulated_oldest_new_audio_latency'] for w in pooled if w['window'] > 0]).item(),
             'first_publication_mean_s': np.mean([r['summary']['first_publication_simulated_seconds'] for r in runs]).item(),
             'max_simulated_queue_wait_s': max(w['simulated_queue_wait'] for w in pooled),
             'final_count': len(first['finals']),
             'text_identical_across_repeats': next(x['text_identical_across_repeats'] for x in aggregate if x['name'] == name),
             'review_flagged_rows': len(first['suspicions'])}
        metrics.append(m)
        detail.extend([f'## {name}: {window}초/{stride}초', '',
                       '| window | 음원 구간(s) | RAW | 조립 match | 새 조각 |',
                       '|---:|---|---|---|---|'])
        for row in first['windows']:
            detail.append(f"| {row['window']} | {row['start']:g}–{row['end']:g} | {row['raw'][row['primary']]} | {row['assembly']['match_type']} | {row['assembly']['new_fragment']} |")
        detail.extend(['', '### FINAL', ''])
        for event in first['finals']:
            detail.append(f"- window {event['window']}, 시간축 {event['stream_time_seconds']:g}s, {event['finalize_reason']}: {event['publication_text']}")
        detail.extend(['', '### 검토 플래그', '',
                       '아래는 자동 휴리스틱이 검토 대상으로 잡은 구간이다. 단어 수정·정상 병합·FINAL 간 이동도 포함하며, 실제 누락/중복 수가 아니다.', ''])
        for suspicion in first['suspicions']:
            detail.append(f"- window {suspicion['window']}: {', '.join(suspicion['reasons'])}; absent ngrams={suspicion.get('absent_ngrams', [])}")
        detail.append('')
        for repeat, result in enumerate(runs):
            for row in result['windows']:
                window_csv.append({'config': name, 'repeat': repeat, 'window': row['window'],
                    'start': row['start'], 'end': row['end'], 'raw': row['raw'][row['primary']],
                    'admitted': row['admitted'][row['primary']], 'assembled': row['assembly']['session_text'],
                    'new_fragment': row['assembly']['new_fragment'], 'match_type': row['assembly']['match_type'],
                    'stt_seconds': row['stt_seconds'], 'server_seconds': row['server_timing']['processing_seconds'],
                    'client_seconds': row['client_seconds'], 'local_pipeline_seconds': row['local_pipeline_seconds'],
                    'simulated_result_time': row['simulated_result_time']})
    save(BASE / 'metrics.json', metrics)
    for name, rows in [('metrics.csv', metrics), ('windows.csv', window_csv)]:
        with (BASE / name).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (BASE / 'RAW_AND_FINAL.md').write_text('\n'.join(detail) + '\n', encoding='utf-8')
    lines = ['| 설정 | window/stride | windows/STT 호출 | 미처리 tail | STT 합계/회(s) | 서버 합계/회(s) | 처리 평균/p95(s) | window 시작 기준 지연(s) | 최초 공개 시각(s) |',
             '|---|---|---|---|---|---|---|---|---|']
    for m in metrics:
        lines.append(f"| {m['name']} | {m['window']}/{m['stride']} | {m['windows_per_run']}/{m['stt_calls_per_run']} | {m['tail_seconds']:g}s | {m['stt_total_mean_s']:.3f} | {m['server_total_mean_s']:.3f} | {m['service_mean_s']:.3f}/{m['service_p95_s']:.3f} | {m['window_start_latency_mean_s']:.3f} | {m['first_publication_mean_s']:.3f} |")
    (BASE / 'MEASUREMENTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print('SOURCE_FILES', integrity['file_count'])
    print('VERSIONS', meta['versions'])
    print('ORDER', meta['orders'])
    print('TEXT_STABILITY', [(m['name'], m['text_identical_across_repeats']) for m in metrics])


if __name__ == '__main__':
    main()
