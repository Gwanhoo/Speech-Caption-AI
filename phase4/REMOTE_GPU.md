# Windows client / RunPod GPU server MVP

The local GPU pipeline remains the default. Remote mode uses the same capture,
sliding windows, bounded queues, speaker tracker, subtitle states and WebSocket UI.
It sends each complete window to the GPU server and consumes raw slot results.

## Responsibility boundary

| Data / processing | Owner | Purpose |
|---|---|---|
| WASAPI, first-channel selection, 48 kHz -> 16 kHz resample | Windows | Original capture |
| 3 s window / 2 s stride, index, capture and stream timestamps | Windows | Audio timeline and backpressure |
| Original window waveform | Windows -> server | Separation input; client retains it for tracking |
| MossFormer2 AMP, finite recovery, silence gate | Server | Two raw separation slots |
| Silero VAD (CPU), faster-whisper large-v3-turbo CUDA FP16 | Server | VAD and transcript for each raw slot |
| Raw separated waveforms | Server -> Windows | Exact overlap waveform used by tracker |
| Raw slot -> logical speaker permutation | Windows | Adjacent-window identity continuity |
| Permuting waveforms, VAD and transcripts together | Windows | Keep evidence attached to the correct speaker |
| Overlap assembler, stable/tentative state, PARTIAL/FINAL | Windows | Continuous per-speaker subtitles |
| WebSocket / browser UI | Windows | Existing local display path |

The separation worker performs the HTTP request. It then invokes the existing
tracker and places a `SeparatedWindow` with the remote result in the existing
separated queue. The downstream worker reads the matching VAD/transcript instead
of running models. The server has no logical speaker tracker or subtitle state.

## HTTP contract (schema_version 1)

`GET /healthz`: HTTP 200 with `status=ready`, `gpu_ready=true`, `models_loaded=true`,
model load count, request counters and GPU memory.

`POST /v1/process`: finite mono 16 kHz WAV, maximum 30 seconds / 16 MiB.
The client writes a FLOAT WAV to retain amplitudes above 1.0 without clipping.

Headers:

- `X-Request-ID`: unique request identity, echoed in response.
- `X-Window-Index`: non-negative window index, echoed in response.
- `X-Capture-Timestamp`: capture timestamp string, echoed unchanged.
- `X-Stream-Start-Seconds`, `X-Stream-End-Seconds`: non-negative stream positions.

The successful JSON includes:

```json
{
  "type": "processing_result",
  "schema_version": 1,
  "request_id": "window-0-...",
  "window_index": 0,
  "capture_timestamp": "2026-09-29T00:00:00.000+00:00",
  "stream_start_seconds": 0.0,
  "stream_end_seconds": 3.0,
  "audio": {"sample_rate": 16000, "channels": 1, "duration_seconds": 3.0},
  "speakers": [
    {
      "raw_slot": 0,
      "sample_count": 48000,
      "waveform": {"encoding": "base64-f32le", "data": "..."},
      "vad": {"speech_detected": true, "speech_duration_ms": 2000,
              "speech_ratio": 0.667, "processing_seconds": 0.05,
              "rms": 0.1, "peak": 1.5, "audio_duration_ms": 3000,
              "timestamps": [{"start": 0, "end": 32000}]},
      "raw_transcript": "한국어 자막",
      "stt_seconds": 0.1
    }
  ],
  "timing": {"queue_wait_seconds": 0.0, "separation_seconds": 0.3,
             "amp_seconds": 0.3, "fp32_seconds": 0.0, "vad_seconds": 0.1,
             "stt_seconds": 0.2, "processing_seconds": 0.6, "end_to_end_rtf": 0.2},
  "fp32_fallback": false,
  "pre_separation_silence": false,
  "gpu_memory": {"device_used_mib": 1023, "process_used_mib": 1014}
}
```

The example abbreviates `speakers`; real responses always contain raw slots 0 and
1, each with exactly the input sample count. `raw_slot` is not a logical speaker
identity. The waveform is contiguous little-endian float32, base64 encoded.
Both 3-second waveforms total about 512 KB in base64. This preserves values above
1.0 (observed in the A5000 validation), unlike unscaled PCM16.

Errors use non-200 HTTP status and
`{"type":"error","request_id":"...","error":{"code":"...","message":"...","retryable":false}}`.
Malformed input is 400, size violations 413, absent routes 404, inference failure
500. The client checks HTTP status, JSON, schema version, identity, slot count,
audio format, sample length, finite waveform, VAD and timing fields.
It raises `RemoteConnectionError`, `RemoteTimeoutError`, `RemoteHTTPError` or
`RemoteProtocolError`, all derived from `RemoteGPUError`.

Connect/read timeouts default to 5/30 seconds. A failed in-flight window is logged
and skipped; capture and later windows continue. Failed indices are not renumbered,
so the tracker recognizes discontinuities. The final JSON reports errors and the
run exits with failure status if any window failed. There is no retry or local
fallback. Startup health failure stops startup with a visible error.

## Execution

The default remote STT model is now `large-v3-turbo`. Select `--whisper-model base`
to reproduce the previous baseline; `small`, `medium`, and `large-v3` are also
supported. `/healthz` reports `stt_model` and `stt_options`. Startup warms
separation, VAD and Whisper in the persistent inference thread before serving;
`--no-warmup` is for cold-start measurement. See the measured limits and accuracy
results in [the 2026-09-30 audit](../ACCURACY_AUDIT.md).

For remote capture diagnostics, use `--save-wav --latency-diagnostics`.
`phase3/output/overlap_debug` contains FLOAT mixed/raw/logical WAVs, preserving
separator peaks above 1.0. Capture JSON now records per-channel levels and
potential downmix cancellation. `--capture-channel first` remains the default;
`mean` is an explicit comparison option.

RunPod / Linux:

```bash
python phase4/gpu_processing_server.py --host 0.0.0.0 --port 8787
```

The default bind is still `127.0.0.1`. Use it for same-host testing. For a Windows
client, use a directly reachable RunPod host and port. No tunnel, domain, TLS or
authentication is implemented in this stage.

Windows PowerShell (Python with the project's dependencies, CPU PyTorch is
sufficient for remote mode; no local CUDA model is loaded):

```powershell
python phase3/run_overlap_pipeline.py `
  --live --duration 60 `
  --processing-mode remote `
  --remote-server-url http://SERVER_ADDRESS:8787 `
  --remote-connect-timeout 5 --remote-read-timeout 30 `
  --assemble --websocket `
  --json-output phase3/output/remote_live_60s.json
```

Open `phase4/web/index.html` on Windows; the existing UI connects to the Windows
WebSocket at `ws://127.0.0.1:8765`.

Local mode:

```powershell
python phase3/run_overlap_pipeline.py --live --duration 60 --processing-mode local --stt-backend sensevoice --vad --assemble --websocket
```

Omitting `--processing-mode` selects local. Existing local STT and VAD flags remain
unchanged. Remote mode always uses the server's faster-whisper and 200 ms Silero
VAD. It rejects `--stt-backend sensevoice`, non-default VAD minimum and
`--diagnostic-audio`; the latter currently requires whole-stream local STT.
`--save-wav` and separation-quality comparison remain available.

Remote window metrics include HTTP roundtrip, server timing and server GPU memory.
Client-local CUDA counters are zero in remote mode; the server memory is reported
separately. Timing is measured on each host with its own monotonic clock; stream
timestamps are client-owned. Windows queue latency includes HTTP/VAD/STT together.

## Tests and recorded validation

```bash
python -m pytest -q phase4/test_remote_gpu_client.py phase4/test_remote_overlap_pipeline.py
python phase4/validate_remote_gpu.py --server-url http://127.0.0.1:8787
```

Unit tests cover schema parsing, waveform restoration, timeout/HTTP/connection
errors, fixed-WAV HTTP processing, tracker/subtitle continuity, actual pipeline
workers with mocked capture, failure isolation and the default local branch.
The worker tests do not open an audio device or perform CUDA inference.

`validate_remote_gpu.py` uses the existing 5-second fixture to produce two
overlapping 3-second windows, followed by silence. It calls the real HTTP server,
tracks speakers, retains assembler/state across windows, and receives PARTIAL and
FINAL events through the existing WebSocket broadcaster. Results are written to
`phase4/output/remote_gpu_integration.json`.

On A5000, the recorded integration passed three windows and six WebSocket events.
Roundtrips were 2.446 / 1.770 / 0.281 seconds; FP32 fallback was absent; VRAM remained
1023 MiB. The first request exceeded the 2-second stride, so initial queue behavior
must be checked on Windows with real capture and network transport.

Windows acceptance checks still required: loopback device/channel, correct 3/2
window sample counts, actual network roundtrip and queue drops, speaker permutation
continuity, silence finalization, browser reception, server interruption/recovery,
and unchanged local GPU execution.
