"""Reproduce the P0-1 synthetic comparison against the audited baseline.

Run at repository root with PYTHONPATH=phase3:phase4:.
No WASAPI, GPU inference, or external network; writes /tmp/p0-comparison.json.
Synthetic capture uses 40ms per read, not the real 3s/2s audio clock.
"""
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import test_remote_overlap_pipeline as tests
from test_remote_gpu_client import payload

current = tests.pipeline
source = Path('/tmp/p0_baseline_pipeline.py')
source.write_bytes(subprocess.check_output(['git', 'show', '9ce4c61:phase3/run_overlap_pipeline.py']))
spec = importlib.util.spec_from_file_location('p0_baseline_pipeline', source)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
with patch.dict(sys.modules, {'soundcard': MagicMock()}):
    spec.loader.exec_module(old)
# ROOT depends on the source path; preserve real repository assets.
old.ROOT = current.ROOT
records = []
for label, module in [('before', old), ('after', current)]:
    tests.pipeline = module
    for scenario in ('normal', 'stop_inflight', 'stale_queued', 'overload'):
        for repeat in range(3):
            case = tests.PipelineTests()
            hooks = module.LivePipelineHooks()
            calls = []
            def response(audio, index):
                calls.append(index)
                if scenario == 'stop_inflight':
                    time.sleep(.2)
                    hooks.stop_event.set()
                elif scenario == 'overload' and index == 0:
                    time.sleep(.5)
                return payload('r', index, len(audio))
            original = module.base.enqueue_or_drop
            def enqueue(target, item, name, index, *args):
                if scenario == 'stale_queued' and name == 'audio_queue' and index == 0:
                    item.capture_ended -= 7
                return original(target, item, name, index, *args)
            started = time.perf_counter()
            with patch.object(module.base, 'enqueue_or_drop', side_effect=enqueue):
                code, result = case.run_pipeline(
                    True, duration=17 if scenario == 'overload' else 15 if scenario == 'stop_inflight' else 5,
                    response_factory=response, runtime_hooks=hooks, speaker_mode='single', websocket=False)
            records.append(dict(version=label, scenario=scenario, repeat=repeat,
                                wall_seconds=time.perf_counter()-started, requests=calls,
                                processed=len(result['windows']), drops=result['dropped'],
                                errors=result['errors'], exit_code=code,
                                cancelled=result.get('cancelled',False),
                                post_capture_seconds=[r['post_capture_latency_seconds'] for r in result['windows']]))
tests.pipeline = current
Path('/tmp/p0-comparison.json').write_text(json.dumps(records, indent=2))
for scenario in ('normal','stop_inflight','stale_queued','overload'):
    for label in ('before','after'):
        rows=[r for r in records if r['scenario']==scenario and r['version']==label]
        print(scenario,label,'wall median',round(statistics.median(r['wall_seconds'] for r in rows),3),
              'requests',[len(r['requests']) for r in rows], 'drops',[len(r['drops']) for r in rows],
              'errors',[r['errors'] for r in rows])
