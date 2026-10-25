"""Compare experiment adapter to actual production workers with saved GPU RAW.

Uses the repository's worker test harness. Only capture, HTTP and device access
are supplied; production tracking/admission/assembler/state stay unmodified.
"""
import base64
import copy
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import soundfile as sf

from compare_windows import CONFIGS, ROOT, Replay, SR, save
from test_remote_overlap_pipeline import PipelineTests, pipeline
from test_remote_gpu_client import payload


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k not in {'matching_seconds', 'audio_overlap_evidence'}}
    if isinstance(value, list):
        return [clean(x) for x in value]
    return value


def main(source):
    audio, _ = sf.read(ROOT / 'phase3/output/evidence/news_20261022_160827/original.wav', dtype='float32')
    checks = []
    for name, window, stride in CONFIGS:
        saved = json.loads((source / f'{name}_repeat_0.json').read_text())
        position = 0
        inputs = []
        def capture(dummy):
            nonlocal position
            # Let production bounded queues drain; this is a logic test, not timing.
            time.sleep(.25)
            value = audio[position:position + len(dummy)].reshape(-1, 1)
            position += len(dummy)
            return value
        def response(chunk, index):
            row = saved['windows'][index]
            expected = audio[int(row['start'] * SR):int(row['end'] * SR)]
            np.testing.assert_array_equal(chunk, expected)
            result = payload('r', index, len(chunk))
            result['pre_separation_silence'] = not np.any(chunk)
            waves = (chunk, np.zeros_like(chunk))
            for i, slot in enumerate(result['speakers']):
                slot['waveform']['data'] = base64.b64encode(waves[i].astype('<f4').tobytes()).decode()
                slot['raw_transcript'] = row['raw'][i]
                slot['vad'] = {k: copy.deepcopy(row.get('raw_vad', row['vad'])[i][k]) for k in (
                    'speech_detected', 'speech_duration_ms', 'speech_ratio', 'rms',
                    'peak', 'processing_seconds', 'timestamps', 'audio_duration_ms')}
            inputs.append((index, row['end'], chunk.copy(), waves, copy.deepcopy(result['speakers']),
                           result['pre_separation_silence']))
            return result
        original_main = pipeline.main
        def main_with_geometry(*, runtime_hooks=None):
            return original_main(sys.argv[1:] + ['--window-seconds', str(window), '--stride-seconds', str(stride)],
                                 runtime_hooks=runtime_hooks)
        harness = PipelineTests()
        with patch.object(pipeline.base, 'CAPTURE_SAMPLE_RATE', SR), patch.object(pipeline, 'main', main_with_geometry):
            code, result = harness.run_pipeline(True, duration=40, websocket=False, speaker_mode='single',
                                                capture_factory=capture, response_factory=response)
        assert code == 0, result['errors']
        assert not result['dropped'], result['dropped']
        assert len(result['windows']) == len(saved['windows'])
        replay = Replay(window, stride)
        events = []
        for data, actual in zip(inputs, result['windows']):
            expected = replay.process(*data)
            assert clean(expected['assembly']) == clean(actual['assembly_events'][0]), (name, data[0], 'assembly')
            assert clean(expected['subtitle_events']) == clean(actual['subtitle_state_events']), (name, data[0], 'state')
            events.extend(expected['subtitle_events'])
        events.extend(replay.flush(inputs[-1][0], inputs[-1][1]))
        assert clean(events) == clean(result['subtitle_state_events']), (name, 'all events including flush')
        checks.append({'config': name, 'windows': len(inputs), 'events': len(events),
                       'assembly_and_state_equal': True,
                       'published_final_count': sum(e['status'] == 'final' and bool(e['publication_text']) for e in events)})
        print(name, 'production-worker parity PASS', checks[-1], flush=True)
    save(Path(__file__).parent / 'pipeline_parity.json', checks)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
