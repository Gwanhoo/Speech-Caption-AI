from __future__ import annotations

import argparse

import run_continuous


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_seconds", type=int, choices=(1, 2))
    parser.add_argument("--tag", choices=("retry",))
    args = parser.parse_args()
    seconds = args.chunk_seconds
    suffix = f"short_{seconds}s" + (f"_{args.tag}" if args.tag else "")

    run_continuous.CHUNK_SECONDS = seconds
    run_continuous.CHUNK_COUNT = 20 // seconds
    run_continuous.CAPTURE_SECONDS = 20
    run_continuous.PHASE_NAME = "Phase 2-E"
    run_continuous.INPUT_DIR = run_continuous.ROOT / "phase2" / "input" / suffix
    run_continuous.OUTPUT_DIR = run_continuous.ROOT / "phase2" / "output" / suffix
    print(f"Phase 2-E condition: {seconds}-second chunks, 20-second capture", flush=True)
    return run_continuous.main()


if __name__ == "__main__":
    raise SystemExit(main())
