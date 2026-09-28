from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from websockets.asyncio.client import connect

sys.path.insert(0, str(Path(__file__).resolve().parent))

from subtitle_websocket import SubtitleWebSocketServer


def event(sequence: int, speaker: str) -> dict[str, object]:
    return {
        "type": "subtitle",
        "sequence": sequence,
        "window_index": sequence - 1,
        "speaker": speaker,
        "text": "WebSocket 자막 테스트입니다.",
        "assembled_text": "WebSocket 자막 테스트입니다.",
        "raw_text": "WebSocket 자막 테스트입니다.",
        "overlap_text": "",
        "start_ms": 0,
        "end_ms": 3000,
        "timestamp": 1_700_000_000_000,
    }


async def receive_one(uri: str, server: SubtitleWebSocketServer, payload: dict[str, object]) -> dict[str, object]:
    async with connect(uri) as websocket:
        result = server.publish(payload)
        assert result.accepted
        received = json.loads(await asyncio.wait_for(websocket.recv(), timeout=2))
        await websocket.send(
            json.dumps(
                {
                    "type": "receipt",
                    "sequence": received["sequence"],
                    "sent_at_ms": received["sent_at_ms"],
                    "received_at_ms": received["sent_at_ms"] + 1,
                }
            )
        )
        return received


def main() -> int:
    server = SubtitleWebSocketServer(port=8766, queue_maxsize=2)
    server.start()
    try:
        first = asyncio.run(receive_one("ws://127.0.0.1:8766", server, event(1, "speaker_0")))
        second = asyncio.run(receive_one("ws://127.0.0.1:8766", server, event(2, "speaker_1")))
        assert first["sequence"] == 1 and first["speaker"] == "speaker_0"
        assert second["sequence"] == 2 and second["speaker"] == "speaker_1"
        assert server.flush()
        stats = server.snapshot()
        assert stats["published"] == 2
        print(json.dumps({"first": first, "second": second, "stats": stats}, ensure_ascii=False))
        return 0
    finally:
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
