"""Run existing tests headlessly; hardware capture is not exercised."""
import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'phase3'), str(ROOT / 'phase4'), str(Path(__file__).parent)]
# Same import-only workaround as phase4/test_remote_overlap_pipeline.py.
# Existing tests inject their own capture/HTTP doubles. No assertions are changed.
sys.modules['soundcard'] = MagicMock(name='unavailable_hardware_capture')


def main():
    results = []
    for directory in ('phase3', 'phase4', 'experiments/r2_b6'):
        suite = unittest.TestLoader().discover(str(ROOT / directory), pattern='test_*.py')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        results.append(result.wasSuccessful())
        print(f'SUITE {directory}: run={result.testsRun} errors={len(result.errors)} '
              f'failures={len(result.failures)} expectedFailures={len(result.expectedFailures)} '
              f'skipped={len(result.skipped)}', flush=True)
    # unittest discovery imports these older assertion scripts without executing main.
    for path in sorted(list((ROOT / 'phase3').glob('test_*.py')) + list((ROOT / 'phase4').glob('test_*.py'))):
        source = path.read_text()
        if '\ndef main(' in source:
            module = importlib.import_module(path.stem)
            code = module.main()
            if code not in (0, None):
                raise RuntimeError(f'{path.name} failed: {code}')
            print(f'ASSERTION_SCRIPT {path.name}: PASS', flush=True)
    return 0 if all(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
