from __future__ import annotations

import argparse
import itertools
import time

from subtitle_websocket import SubtitleWebSocketServer


def main() -> int:
    parser = argparse.ArgumentParser(description="Send synthetic subtitle events to the Phase 4-A WebSocket UI")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    server = SubtitleWebSocketServer(port=args.port)
    server.start()
    try:
        for sequence in range(1, args.count + 1):
            speaker_index = (sequence - 1) % 2
            text = (
                "WebSocket 자막 테스트입니다."
                if speaker_index == 0
                else "두 번째 화자의 자막 테스트입니다."
            )
            server.publish(
                {
                    "type": "subtitle",
                    "sequence": sequence,
                    "window_index": sequence - 1,
                    "speaker": f"speaker_{speaker_index}",
                    "text": text,
                    "assembled_text": text,
                    "raw_text": text,
                    "overlap_text": "",
                    "start_ms": (sequence - 1) * 2000,
                    "end_ms": (sequence - 1) * 2000 + 3000,
                    "timestamp": int(time.time() * 1000),
                }
            )
            time.sleep(args.interval)
        server.flush()
        print(server.snapshot())
        return 0
    finally:
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
