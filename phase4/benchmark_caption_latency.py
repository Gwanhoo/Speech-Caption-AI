"""Paced WAV -> bounded queue -> real HTTP GPU -> subtitles -> real WebSocket.

This measures a local Linux replay. Capture/device/proxy/Qt rendering remain
outside the measurement. Audio availability follows the production 3s/2s clock.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase3"))
from benchmark_accuracy import distribution
from remote_gpu_client import RemoteGPUClient
from secondary_leakage_diagnostics import (
    build_window_diagnostic,
    secondary_transcript_suppression_reason,
)
from speaker_tracking import PersistentSpeakerTracker
from subtitle_assembler import SubtitleAssembler, SpeakerSubtitleState
from subtitle_websocket import SubtitleWebSocketServer


async def measure(args):
    prepared = json.loads(args.prepared.read_text())
    client = RemoteGPUClient(args.server_url)
    websocket = SubtitleWebSocketServer(port=0)
    websocket.start()
    report = {
        "environment": "Linux local HTTP and WebSocket; WAV paced capture, no WASAPI/Qt/proxy",
        "health": await asyncio.to_thread(client.health),
        "samples": [],
    }
    try:
        port = websocket._server.sockets[0].getsockname()[1]
        async with connect(f"ws://127.0.0.1:{port}") as browser:
            for sample in args.samples:
                rows = prepared[sample]["windows"]
                queue = asyncio.Queue(maxsize=3)
                tracker = PersistentSpeakerTracker(overlap_samples=16000)
                assemblers = [SubtitleAssembler(i) for i in (0, 1)]
                states = [SpeakerSubtitleState(i) for i in (0, 1)]
                record = {
                    "sample": sample,
                    "windows": [],
                    "events": [],
                    "dropped": [],
                    "max_queue_depth": 0,
                }
                start = time.perf_counter()

                async def producer():
                    for i in range(len(rows) + 2):
                        audio = (
                            sf.read(ROOT / rows[i]["paths"][0], dtype="float32")[0]
                            if i < len(rows)
                            else np.zeros(48000, dtype=np.float32)
                        )
                        due = start + 3 + 2 * i
                        await asyncio.sleep(max(0, due - time.perf_counter()))
                        item = (i, audio, due, time.perf_counter())
                        try:
                            queue.put_nowait(item)
                        except asyncio.QueueFull:
                            record["dropped"].append(i)
                        record["max_queue_depth"] = max(
                            record["max_queue_depth"], queue.qsize()
                        )
                    await queue.put(None)

                task = asyncio.create_task(producer())
                try:
                    while (item := await queue.get()) is not None:
                        i, audio, due, released = item
                        dequeued = time.perf_counter()
                        result = await asyncio.to_thread(
                            client.process, audio, i, "paced-wav", i * 2, i * 2 + 3
                        )
                        assignment = tracker.assign(
                            window=i,
                            raw_speakers=result.raw_speakers,
                            mixture=audio,
                            pre_separation_silence=result.pre_separation_silence,
                        )
                        slots = result.logical_slots(assignment.diagnostic)
                        diagnostic = build_window_diagnostic(
                            speaker_0=assignment.speakers[0],
                            speaker_1=assignment.speakers[1],
                            vad_results=[s["vad"] for s in slots],
                            transcripts=[s["raw_transcript"] for s in slots],
                        )
                        suppression = secondary_transcript_suppression_reason(
                            diagnostic, slots[1]["raw_transcript"]
                        )
                        assembly_started = time.perf_counter()
                        events = []
                        for speaker, slot in enumerate(slots):
                            active = slot["vad"]["speech_detected"] and not (
                                speaker == 1 and suppression
                            )
                            text = slot["raw_transcript"] if active else ""
                            assembly = assemblers[speaker].process(i, text)
                            for event in states[speaker].process(
                                i,
                                assembly.utterance_hypothesis,
                                bool(active),
                                i * 2 + 3,
                                confirmed_prefix_length=assembly.confirmed_prefix_length,
                            ):
                                events.append(event)
                                if event.status == "final":
                                    assemblers[speaker].reset_utterance()
                        assembly_seconds = time.perf_counter() - assembly_started
                        receives = []
                        for event in events:
                            payload = {
                                **event.to_dict(),
                                "type": "subtitle",
                                "speaker": f"speaker_{event.speaker}",
                                "sequence": len(record["events"]) + 1,
                                "timestamp": round(time.time() * 1000),
                            }
                            sent = time.perf_counter()
                            if not websocket.publish(payload).accepted:
                                raise RuntimeError("WebSocket queue rejected event")
                            received = json.loads(
                                await asyncio.wait_for(browser.recv(), 5)
                            )
                            ended = time.perf_counter()
                            if received["text"] != event.text:
                                raise AssertionError("WebSocket text mismatch")
                            receives.append(ended - due)
                            record["events"].append(
                                {
                                    **payload,
                                    "websocket_delivery_seconds": ended - sent,
                                    "capture_ready_to_receive_seconds": ended - due,
                                    "stream_start_to_receive_seconds": ended - start,
                                }
                            )
                        record["windows"].append(
                            {
                                "window": i,
                                "queue_wait_seconds": dequeued - released,
                                "capture_ready_to_result_seconds": time.perf_counter()
                                - due,
                                "request_seconds": result.request_seconds,
                                "server": result.timing,
                                "assembler_seconds": assembly_seconds,
                                "subtitle_latency": receives,
                            }
                        )
                    await task
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                record["transcripts"] = [" ".join(s.final_segments) for s in states]
                report["samples"].append(record)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
                print(
                    sample,
                    "windows",
                    len(record["windows"]),
                    "drops",
                    record["dropped"],
                    flush=True,
                )
        report["websocket"] = websocket.snapshot()
        report["health_after"] = await asyncio.to_thread(client.health)
        report["capture_ready_to_receive_seconds"] = distribution(
            [
                e["capture_ready_to_receive_seconds"]
                for s in report["samples"]
                for e in s["events"]
            ]
        )
        report["request_seconds"] = distribution(
            [w["request_seconds"] for s in report["samples"] for w in s["windows"]]
        )
        report["success"] = True
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    finally:
        client.close()
        websocket.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8788")
    parser.add_argument(
        "--prepared",
        type=Path,
        default=ROOT / "phase3/output/accuracy_audit/prepared.json",
    )
    parser.add_argument(
        "--samples", nargs="+", default=["single_A", "single_B", "mixture"]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "phase3/output/accuracy_audit/latency_small.json",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(measure(args))


if __name__ == "__main__":
    main()
