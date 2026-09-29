"""Fixed-WAV HTTP -> tracker -> continuous subtitle -> WebSocket validation.

Run on Linux without importing soundcard or executing Windows capture.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase3"))
from remote_gpu_client import RemoteGPUClient
from speaker_tracking import PersistentSpeakerTracker
from subtitle_assembler import SubtitleAssembler, SpeakerSubtitleState
from subtitle_websocket import SubtitleWebSocketServer


async def validate(server_url: str) -> dict:
    client = RemoteGPUClient(server_url)
    websocket = SubtitleWebSocketServer(port=0)
    websocket.start()
    tracker = PersistentSpeakerTracker(overlap_samples=16000)
    assemblers = [SubtitleAssembler(i) for i in (0, 1)]
    states = [SpeakerSubtitleState(i) for i in (0, 1)]
    report = {"server_url": server_url, "windows": [], "received_events": []}
    try:
        report["health"] = await asyncio.to_thread(client.health)
        # Ephemeral WebSocket port keeps this validation independent of the UI port.
        port = websocket._server.sockets[0].getsockname()[1]
        audio, sr = sf.read(ROOT / "phase2/input/chunk_000_mixed_16k.wav", dtype="float32")
        assert sr == 16000
        async with connect(f"ws://127.0.0.1:{port}") as browser:
            for window in range(3):
                mixture = audio[window * 32000:window * 32000 + 48000] if window < 2 else np.zeros(48000, dtype=np.float32)
                result = await asyncio.to_thread(client.process, mixture, window, "fixed-wav", window * 2, window * 2 + 3)
                assignment = tracker.assign(window=window, raw_speakers=result.raw_speakers,
                                            mixture=mixture, pre_separation_silence=result.pre_separation_silence)
                slots = result.logical_slots(assignment.diagnostic)
                events = []
                for speaker, slot in enumerate(slots):
                    assembly = assemblers[speaker].process(window, slot["raw_transcript"])
                    for event in states[speaker].process(window, assembly.utterance_hypothesis,
                                                         slot["vad"]["speech_detected"], window * 2 + 3):
                        event_payload = event.to_dict()
                        event_payload.update(type="subtitle", speaker=f"speaker_{speaker}", window_index=window,
                                             sequence=len(report["received_events"]) + 1)
                        assert websocket.publish(event_payload).accepted
                        received = json.loads(await asyncio.wait_for(browser.recv(), 5))
                        assert received["text"] == event.text
                        report["received_events"].append(received)
                        events.append(event.to_dict())
                        if event.status == "final": assemblers[speaker].reset_utterance()
                report["windows"].append({
                    "window": window, "request_seconds": result.request_seconds,
                    "timing": result.timing, "gpu_memory": result.gpu_memory,
                    "fp32_fallback": result.fp32_fallback, "silence_gate": result.pre_separation_silence,
                    "assignment": assignment.diagnostic,
                    "speakers": [{"raw_slot": slot["raw_slot"], "vad": slot["vad"],
                                  "transcript": slot["raw_transcript"],
                                  "rms": float(np.sqrt(np.mean(wave * wave))),
                                  "peak": float(np.max(np.abs(wave)))}
                                 for slot, wave in zip(result.slots, result.raw_speakers)],
                    "subtitle_events": events,
                })
        assert report["windows"][2]["silence_gate"]
        assert any(e["status"] == "partial" for e in report["received_events"])
        assert any(e["status"] == "final" for e in report["received_events"])
        report["health_after"] = await asyncio.to_thread(client.health)
        report["websocket"] = websocket.snapshot()
        report["success"] = True
        return report
    finally:
        client.close()
        websocket.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8787")
    parser.add_argument("--json-output", type=Path, default=ROOT / "phase4/output/remote_gpu_integration.json")
    args = parser.parse_args()
    report = asyncio.run(validate(args.server_url))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"success": report["success"], "windows": len(report["windows"]),
                      "websocket_events": len(report["received_events"]), "json_output": str(args.json_output)}, ensure_ascii=False))


if __name__ == "__main__": main()
