from __future__ import annotations

import argparse
import json
import locale
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "phase3" / "run_realtime_pipeline.py"
DEFAULT_OUTPUT_DIR = ROOT / "phase3" / "output" / "chunk_benchmark"
CHUNK_SIZES = (1, 2, 3, 5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3-C chunk latency comparison")
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0 or any(args.duration % seconds for seconds in CHUNK_SIZES):
        parser.error("--duration must be positive and divisible by 1, 2, 3, and 5")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    exit_code = 0
    for seconds in CHUNK_SIZES:
        result_path = args.output_dir / f"result_{seconds}s.json"
        log_path = args.output_dir / f"run_{seconds}s.log"
        if args.analyze_only:
            if not result_path.is_file():
                print(f"Missing result: {result_path}", flush=True)
                exit_code = 1
                continue
            summaries.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        command = [
            sys.executable,
            "-u",
            str(PIPELINE),
            "--chunk-seconds",
            str(seconds),
            "--duration",
            str(args.duration),
            "--json-output",
            str(result_path),
            "--analyze-separation-quality",
        ]
        print(f"\n========== Running {seconds}s chunks ==========" , flush=True)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding=locale.getpreferredencoding(False),
            errors="replace",
        )
        lines = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        return_code = process.wait()
        log_path.write_text("".join(lines), encoding="utf-8")
        if not result_path.is_file():
            print(f"Condition {seconds}s failed with exit code {return_code}", flush=True)
            exit_code = 1
            continue
        summary = json.loads(result_path.read_text(encoding="utf-8"))
        summaries.append(summary)
        if return_code:
            print(f"Condition {seconds}s completed with FAIL status", flush=True)
            exit_code = 1

    comparison_path = args.output_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps({"duration_seconds": args.duration, "conditions": summaries}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nComparison: {comparison_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
