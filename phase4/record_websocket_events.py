from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException


async def collect(uri: str, expected: int, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    events: list[dict[str, Any]] = []
    reconnects = 0
    while len(events) < expected and time.monotonic() < deadline:
        try:
            async with connect(uri, open_timeout=1) as websocket:
                while len(events) < expected and time.monotonic() < deadline:
                    remaining = max(0.1, deadline - time.monotonic())
                    payload = json.loads(await asyncio.wait_for(websocket.recv(), timeout=remaining))
                    if payload.get("type") != "subtitle":
                        continue
                    received_at_ms = int(time.time() * 1000)
                    payload["browser_received_at_ms"] = received_at_ms
                    events.append(payload)
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "receipt",
                                "sequence": payload["sequence"],
                                "sent_at_ms": payload["sent_at_ms"],
                                "received_at_ms": received_at_ms,
                            }
                        )
                    )
        except (OSError, WebSocketException, asyncio.TimeoutError):
            reconnects += 1
            await asyncio.sleep(0.2)

    sequences = [event["sequence"] for event in events]
    duplicates = len(sequences) - len(set(sequences))
    out_of_order = sum(right <= left for left, right in zip(sequences, sequences[1:]))
    missing = [sequence for sequence in range(1, max(sequences, default=0) + 1) if sequence not in sequences]
    latencies = [
        max(0, event["browser_received_at_ms"] - event["sent_at_ms"]) for event in events
    ]
    return {
        "received_count": len(events),
        "sequences": sequences,
        "duplicates": duplicates,
        "out_of_order": out_of_order,
        "missing": missing,
        "reconnect_attempts": reconnects,
        "browser_receive_latency_ms": {
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "events": events,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Record WebSocket subtitle events for Phase 4-A E2E validation")
    parser.add_argument("--uri", default="ws://127.0.0.1:8765")
    parser.add_argument("--expected", type=int, default=28)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(collect(args.uri, args.expected, args.timeout))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "events"}))
    return 0 if result["received_count"] == args.expected and not result["duplicates"] and not result["out_of_order"] and not result["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
