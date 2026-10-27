"""Run main()-style regressions missed by unittest discovery, headlessly."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
env = dict(os.environ, PYTHONPATH='phase3:phase4:.')
bootstrap = ('import runpy,sys; from unittest.mock import MagicMock; '
             'sys.modules["soundcard"]=MagicMock(); '
             'runpy.run_path(sys.argv[1],run_name="__main__")')
failed = False
for name in ('stt_context', 'subtitle_state', 'subtitle_contract_p1',
             'subtitle_consensus', 'subtitle_assembler',
             'subtitle_assembler_state_integration', 'overlap_diagnostic_schema',
             'pre_separation_silence_gate', 'secondary_leakage_diagnostics'):
    path = f'phase3/test_{name}.py'
    result = subprocess.run([sys.executable, '-c', bootstrap, path], cwd=ROOT,
                            env=env, text=True, capture_output=True)
    print(path, 'PASS' if result.returncode == 0 else 'FAIL', flush=True)
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    if result.stderr.strip():
        print(result.stderr.strip(), flush=True)
    failed |= result.returncode != 0
raise SystemExit(int(failed))
