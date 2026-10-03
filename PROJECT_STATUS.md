# Speech-Caption-AI 인수인계 및 현재 상태

조사 기준: 2026-10-03. 코드 기준은 `main`의 `d2efaad`
(`fix: suppress single-speaker secondary leakage`)다.

이 상단 섹션이 현재 인수인계 기준이다. 현재 소스, deterministic 회귀 테스트,
최근 git history, 기존 Markdown을 대조해 작성했다. 실제 Windows WASAPI/RunPod GPU
Live 실행은 이번 조사에서 수행하지 않았다. 최신 단일화자·두 화자 Live 관찰은
사용자가 제공한 실제 실행 로그의 사실이며, 코드 테스트 통과와 구분한다.

## 프로젝트 목적과 아키텍처

이 프로젝트의 목적은 단순 A/B 화자 표시가 아니다. Windows 시스템 오디오를 받아
음성을 분리·정제하고, 중복을 줄인 읽기 쉬운 한국어 실시간 자막을 PySide6 feed와
transparent overlay에 제공하는 것이다.

```text
Windows system audio (YouTube / Discord / game)
  -> WASAPI loopback, 48 kHz capture
  -> 16 kHz mono, 3 s window / 2 s stride / 1 s overlap
  -> HTTP to RunPod GPU
  -> ClearVoice MossFormer2_SS_16K (two raw separation slots)
  -> Silero VAD and faster-whisper on the server
  -> Windows logical-speaker tracking and subtitle admission
  -> SubtitleAssembler / SpeakerSubtitleState
  -> PARTIAL / FINAL structured event -> local WebSocket
  -> PySide6 chronological feed and transparent overlay
```

Remote mode is the intended Windows production path. The GPU server has no
logical speaker identity or subtitle state. It returns two raw separated
waveforms with VAD/STT; the Windows client owns timeline, tracking, subtitle
admission, assembly, publication, and UI delivery.

## 주요 파일과 실행 경계

| 영역 | 주요 파일 | 현재 역할 |
|---|---|---|
| Live orchestration | `phase3/run_overlap_pipeline.py` | WASAPI, queues, remote/local mode, tracking, state events, WebSocket publication |
| Recovery | `phase3/separation_recovery.py` | finite-input, silence gate, AMP/FP32 recovery |
| Tracking | `phase3/speaker_tracking.py` | raw slot -> logical speaker continuity |
| Admission | `phase3/secondary_leakage_diagnostics.py` | source evidence, leakage diagnostics/routing/admission, weak-tail handling |
| Subtitle state | `phase3/subtitle_assembler.py` | overlap de-duplication, stable/tentative, source-supported PARTIAL/FINAL |
| GPU server | `phase4/gpu_processing_server.py` | persistent MossFormer2, Silero VAD, faster-whisper |
| Remote contract | `phase4/remote_gpu_protocol.py`, `remote_gpu_client.py` | HTTP WAV request/response validation |
| WebSocket | `phase4/subtitle_websocket.py` | bounded local event broadcaster |
| Windows UI | `phase5/gui_app.py`, `live_caption_controller.py`, `subtitle_presentation.py`, `subtitle_overlay.py` | RunPod/WASAPI preflight, Qt worker/controller, feed/overlay |

The raw-slot/logical-speaker boundary is deliberate and must remain on the
Windows client. The server's `raw_slot` value is not a speaker ID.

### Runtime entry points

- Desktop app: `python -m phase5.gui_app`
- Headless Windows pipeline: `python phase3/run_overlap_pipeline.py --live ...`
- RunPod server: `python phase4/gpu_processing_server.py --host 0.0.0.0 --port 8787`
- Real server validation: `python phase4/validate_remote_gpu.py --server-url http://HOST:8787`
- Browser MVP: open `phase4/web/index.html` after local WebSocket starts.

## Environment and launch

`requirements.txt` targets a RunPod PyTorch image and intentionally omits
PyTorch. Install CUDA-compatible PyTorch first on the GPU server, then the
project requirements. The only application environment variable found in
current source is `GPU_SERVER_URL`; the GUI uses it as its server URL default.
Set it for each deployment rather than relying on the code fallback URL.

RunPod/Linux:

```bash
python -m pip install -r requirements.txt
python phase4/gpu_processing_server.py --host 0.0.0.0 --port 8787
```

Windows GUI:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:GPU_SERVER_URL = "https://YOUR-RUNPOD-8787.proxy.runpod.net"
.\.venv\Scripts\python.exe -m phase5.gui_app
```

Windows remote CLI run:

```powershell
.\.venv\Scripts\python.exe phase3\run_overlap_pipeline.py `
  --live --duration 60 --processing-mode remote `
  --remote-server-url "$env:GPU_SERVER_URL" `
  --assemble --websocket --save-wav --latency-diagnostics `
  --json-output phase3\output\remote_live_60s.json
```

The server default is faster-whisper `small`; `--whisper-model` also accepts
`base`, `medium`, `large-v3-turbo`, and `large-v3`. This is not authorization
to change models in the next task. Remote mode forces server-side VAD and
faster-whisper. There is no server authentication, TLS termination, retry, or
local inference fallback; failed windows are logged/skipped while later windows
continue.

## Implemented subtitle lifecycle

The actual downstream order is:

```text
server VAD/STT -> PersistentSpeakerTracker -> resolve_subtitle_fragments()
  -> admit_subtitle_streams() -> SubtitleAssembler.process()
  -> SpeakerSubtitleState.process() -> publish_subtitle_state_event()
  -> WebSocket/runtime hook -> Qt presentation
```

`SubtitleAssembler` handles overlap/revision seams. `SpeakerSubtitleState`
holds a complete per-logical-speaker hypothesis, stable/tentative consensus, and
`_source_supported_text`. In production `require_final_support=True`,
`publication_text` comes from `_source_supported_text`, not merely from
`stable_text`. The publisher sends only nonempty publication text, or a
retraction for an already published utterance. The Qt presentation layer only
renders structured events; it does not create a secondary stream itself.

On silence, a validated utterance may FINAL using retained
`source_supported_text`; an unsupported tentative becomes an internal `discard`
with empty publication text. This distinction is intentional and must survive
the next lifecycle change.

`d2efaad` added owner/candidate acoustic diagnostics, an internal hold for some
owner-like secondary candidates, structural cross-stream overlap checks, and
`existing_partial_weak_unlocalized_tail`. These changes have deterministic
coverage, but do not prove the latest Live behavior is correct.

## Current core blocker: secondary candidate lifecycle

Two opposite Live failures remain and should be treated as one provenance/state
machine issue, not as a Whisper, MossFormer, or threshold-only task.

### A. Single speaker: false secondary is admitted and published

One news recording was played once; there was no second speaker. In window 003,
the secondary nevertheless reached a visible subtitle row:

```text
reason=independent_input_support
source_supported=true
will_publish=true
owner_input_correlation ~= 0.9874
candidate_input_correlation ~= 0.9144
independent_input_correlation ~= 0.9002
input_residual_energy_fraction ~= 0.02498
candidate_residual_energy_fraction ~= 0.27718
owner_candidate_conflict=true
speech_contained_in_owner=true
```

Primary and secondary represented the same original speech but were transcribed
as different strings. Exact lexical equality therefore cannot be a mandatory
first-window leakage test. The following window kept the false secondary through
`existing_partial`. The weak tail itself was successfully classified later as
`existing_partial_weak_unlocalized_tail`, but that does not repair the earlier
false admission.

### B. Real two speakers: genuine secondary never promotes

Two different news sources played concurrently produced long, semantically
distinct streams: one Ukraine/North-Korean-POW/Korea/presidential-office news,
the other 2030/housing-subscription/deposit/loan-interest news. Separation and
server STT were therefore useful in this test.

The genuine secondary first showed:

```text
reason=speech_outside_owner_awaiting_confirmation
source_support_deferred=true
VAD ratio ~= 0.838
speech duration ~= 2514 ms
independent_input_correlation ~= 0.999
speech_contained_in_owner=false
owner_candidate_conflict=false
```

Later windows kept producing independent STT and `source_supported=true`, yet
state/publication remained `publication_text=""`, `will_publish=false`,
`reason=awaiting_source_support`. Silence finally produced
`discarded_unsupported_tentative`.

### Required next state transition

`existing_partial` must not itself mean that a secondary candidate is validated.
The next code change must introduce the smallest explicit provenance distinction:

```text
NEW SECONDARY -> TENTATIVE -> re-evaluate on the next overlap window

single-speaker leakage:
  TENTATIVE -> repeated owner/leakage evidence -> DISCARD
  -> never source_supported_text -> no WebSocket PARTIAL/FINAL

independent second speaker:
  TENTATIVE -> sustained independent acoustic evidence -> CONFIRMED/VALIDATED
  -> source_supported_text -> normal PARTIAL/FINAL publication
```

Do not solve this with phrase blacklists, blanket secondary blocking, a
correlation threshold nudge, first-window lexical equality, MossFormer redesign,
or a Whisper model change. Preserve short legitimate second-speaker replies.

## Next work order and acceptance criteria

1. Add explicit secondary tentative-versus-validated provenance and deterministic
   regressions using the latest single-speaker and two-news evidence.
2. Run the Live acceptance matrix below; only then freeze admission/separation.
3. Compare STT models on fixed original/separated WAVs: faster-whisper `small`
   baseline versus `large-v3-turbo` and suitable Korean candidates, using CER/WER,
   numbers, proper nouns, phonetic confusions, latency, and VRAM.
4. Stabilize presentation/overlay, package Windows EXE, freeze the demo, prepare
   the graduation presentation.

| Scenario | Required result |
|---|---|
| Same single news, at least three runs | visible primary; no false secondary row, WebSocket publish, or silence FINAL ghost |
| Current two-news overlap | both streams visible; genuine tentative promotes after sustained evidence |
| Confirmation latency | roughly one stride, about 2 seconds; never permanent `awaiting_source_support` |
| Short true reply | not permanently treated as leakage |
| Weak tail | retain `existing_partial_weak_unlocalized_tail` behavior |
| Silence | discard unconfirmed tentative; preserve FINAL for validated/published text; no new ghost |

## Diagnostics required for the next Live run

Keep the complete JSON and console logs for matching windows. Correlation fields
may be nested in `speech_local_evidence` and `source_support_checks`.

| Log family | Required fields |
|---|---|
| `SUBTITLE ADMISSION` | `reason`, `suppressed`, `existing_partial`, `source_supported`, `source_support_deferred`, `owner_candidate_conflict`, owner/candidate/independent correlations, both residual fractions, `speech_contained_in_owner`, `cross_stream_lexical_overlap`, `weak_speech_check_applied`, `weak_existing_tail_suppressed` |
| `SUBTITLE SUPPORT` | `source_text`, `retained_prefix_length`, `observed_suffix_matches`, `unobserved_prefix_length`, `unsupported_gap_length`, `reason` |
| `CONSENSUS` | `stable`, `tentative`, `action`, `history` |
| `SUBTITLE PUBLICATION` | `source_supported`, `source_supported_text`, `publication_text`, `will_publish`, `reason`, `needs_retraction` |
| Published event | `sequence`, logical `speaker`, `utterance_id`, `status`, `action` |

For false secondary, confirm no secondary WebSocket sequence exists. For genuine
secondary, find the exact promotion window where `source_supported_text` becomes
nonempty and the following PARTIAL publishes. A valid primary FINAL may have
`source_supported=false` in the silence window while retaining nonempty
`source_supported_text`; that is expected.

## Tests and current status

The following command was executed for this handoff:

```bash
PYTHONPATH=/tmp/speech-caption-pytest-only:phase3:phase4:phase5:. \
python -m pytest -q . \
  --ignore=phase1/diagnostic/wasapi_test.py \
  --ignore=phase3/run_secondary_validity_live_test.py \
  --ignore=phase3/test_overlap_diagnostic_schema.py \
  --ignore=phase3/test_secondary_validity_live_test.py \
  --ignore=phase5/test_gui_app.py \
  --ignore=phase5/test_ui_shell.py
```

Result: **126 passed, 3 xfailed, 95 subtests passed**.

The three xfails in `phase3/test_source_support_diagnostics.py` remain visible:
unresolved gap hiding later support, noise above the existing bound, and a
projection artifact publication case. They are not successes. Excluded files
need local audio/PySide6 runtime support, so this result is not Windows capture,
RunPod endpoint, or GUI-rendering validation.

Important tests:

| Area | Files |
|---|---|
| Admission/lifecycle | `phase3/test_ghost_subtitle_regressions.py`, `test_subtitle_admission.py`, `test_source_support_diagnostics.py`, `test_subtitle_finalization_evidence.py`, `test_subtitle_consensus.py` |
| Assembly/state | `test_subtitle_assembler.py`, `test_subtitle_assembler_state_integration.py`, `test_subtitle_state.py`, `test_live_caption_continuity.py` |
| Tracking/recovery | `test_speaker_tracking.py`, `test_secondary_leakage_diagnostics.py`, `test_separation_recovery.py`, `test_pre_separation_silence_gate.py` |
| Remote/WebSocket | `phase4/test_gpu_processing_server.py`, `test_remote_gpu_client.py`, `test_remote_overlap_pipeline.py`, `test_websocket_mvp.py` |
| Qt presentation | `phase5/test_subtitle_presentation.py`, `test_gui_app.py`, `test_ui_shell.py` |

## Technical debt and guardrails

- Deterministic fixtures do not yet model the newest Live lexical-divergence
  leakage case or the true-two-speaker promotion failure.
- Current source support retains text prefixes but has no explicit persisted
  secondary `tentative` / `validated` provenance model.
- STT accuracy lacks a human-verified Korean reference corpus.
- Long uninterrupted speech, music/game effects, short replies, WASAPI channel
  choice, poor network conditions, long-run recovery, deployment auth/TLS, and
  Windows GUI rendering remain incompletely proven.

Do not redesign these completed parts: the client/server boundary; 48 kHz ->
16 kHz conversion; 3 s/2 s overlap cadence; server MossFormer2 -> VAD -> STT
ordering; Windows logical-speaker tracking; assembler seam handling;
source-supported FINAL safety; weak-tail suppression; or silence finalization
for validated subtitles. Do not remove diagnostic fields or `d2efaad` fixtures.

## Reading older documents

`phase4/REMOTE_GPU.md` remains the protocol/launch reference. `phase3/
LIVE_SOURCE_SUPPORT_DIAGNOSIS.md` explains earlier diagnostics and `d2efaad`
fixtures but predates the newest Live observations above. `ACCURACY_AUDIT.md`
is useful historical model/latency evidence, not a replacement for the pending
fixed-WAV comparison. `README.md` still says in one historical section that VAD
and speaker tracking are unimplemented; current code implements Silero VAD and
`PersistentSpeakerTracker`. No `REMAINING_WORK_GUIDE.md` exists in this checkout.

## Historical snapshot: superseded 2026-09-27 document

The remainder is retained only as project history. It contains stale claims,
including a SenseVoice/browser-centered status. Do not use it for current
implementation decisions; use the sections above and current `main` code.

<details>
<summary>Superseded 2026-09-27 snapshot (historical reference only)</summary>

## 1. 프로젝트 최종 목표

이 프로젝트는 Windows 시스템 또는 웹 오디오를 받아 두 사람이 동시에 말하는 상황에서도 화자별 음성을 분리하고, 각각의 음성을 한국어 자막으로 표시하는 청각장애인 보조용 실시간 자막 시스템을 만드는 프로젝트다.

목표 데이터 흐름은 다음과 같다.

```text
Windows 시스템/웹 오디오
  -> WASAPI loopback
  -> 연속 audio window
  -> MossFormer2 음성 분리
  -> speaker_0 / speaker_1
  -> Silero VAD
  -> STT
  -> overlap assembler
  -> subtitle stabilization / multi-window consensus
  -> PARTIAL / FINAL 이벤트
  -> WebSocket
  -> 웹 브라우저 자막 UI
```

현재 저장소에는 위 흐름의 실행 가능한 오프라인, 테스트 WAV, Live, WebSocket, 브라우저 UI 경로가 모두 존재한다. 다만 STT 출력 품질과 실제 화자 분리 품질은 아직 제품 수준으로 확정되지 않았다.

## 2. 현재 프로젝트 구조

```text
phase0/   오프라인 두 WAV mixing -> separation -> STT 최소 검증
phase1/   Windows WASAPI loopback capture 및 채널/오디오 진단
phase2/   MossFormer2 CPU/GPU, AMP, chunk, 연속 처리 성능 실험
phase3/   실시간 capture -> separation -> STT, chunk/overlap/자막 안정화
phase4/   WebSocket broadcaster, 브라우저 자막 UI MVP, WebSocket 테스트
checkpoints/  MossFormer2 및 faster-whisper 로컬 모델 파일
```

### 현재 실행의 중심

- `phase3/run_overlap_pipeline.py`: 현재 기준 실행 진입점. WASAPI loopback, 3초 window/2초 stride, bounded queue, MossFormer2, VAD, STT, assembler, subtitle state, WebSocket을 한 프로세스로 연결한다.
- `phase3/separation_recovery.py`: 입력 finite 검사, AMP inference, non-finite 출력 검출, FP32 fallback, pre-separation silence gate를 제공한다.
- `phase3/run_realtime_pipeline.py`: Phase 3-B 계열의 1/2/3/5초 chunk 파이프라인과 모델 로딩/STT/VAD 공용 코드가 있는 기반 모듈이다.
- `phase3/subtitle_assembler.py`: overlap 중복 제거, PARTIAL/FINAL 상태, stable/tentative consensus 및 silence finalization을 구현한다.
- `phase4/subtitle_websocket.py`: 별도 thread/async WebSocket broadcaster와 bounded event queue를 구현한다.
- `phase4/web/index.html`, `phase4/web/app.js`, `phase4/web/styles.css`: 두 speaker의 incremental subtitle event를 표시하는 로컬 UI다.

초기 README에는 “VAD, Speaker Tracking은 아직 구현하지 않음”이라는 설명이 남아 있지만, 현재 실제 코드에는 VAD와 subtitle stabilization/consensus가 구현되어 있다. Speaker embedding, diarization, 장기 speaker tracking은 현재 코드에서 확인되지 않는다.

## 3. 현재 사용 기술 및 모델

### Audio capture

- `soundcard`의 Windows WASAPI loopback을 사용한다.
- `phase3/run_overlap_pipeline.py`는 `sc.default_speaker()`와 `sc.get_microphone(speaker.name, include_loopback=True)`를 사용한다.
- capture rate는 기반 코드의 `CAPTURE_SAMPLE_RATE = 48_000`이다.
- 캡처 데이터는 현재 `raw[:, 0]` 첫 채널을 사용해 16 kHz mono로 resample한다. 모든 WASAPI 채널을 평균 downmix하는 구조가 아니다.
- 기본 overlap pipeline은 3초 window, 2초 stride, 1초 overlap이다.
- 각 window 이후 stride 구간만 새로 캡처하고 retained tail과 합쳐 3초 window를 만든다.

### Speech separation

- ClearVoice의 `MossFormer2_SS_16K`를 사용한다.
- 입력은 16 kHz mono NumPy `float32`; 추론 시 `(1, samples)` 형태로 batch를 만든다.
- 모델은 CUDA에 로드되고 separation에는 `torch.inference_mode()`와 CUDA `autocast(float16)`가 적용된다.
- AMP 출력에 NaN/Inf가 있으면 같은 입력을 FP32로 재시도한다.
- FP32도 non-finite이면 해당 window만 실패 처리하고 오류/metrics에 기록한다.
- finite input 검사 후 `RMS <= 1e-5`이면 pre-separation silence gate가 동작한다. MossFormer2를 호출하지 않고 `(2, 1, samples)` 형태의 zero `float32` 두 채널을 만들어 기존 VAD/queue 경로로 보낸다.
- gate 기준은 `phase3/separation_recovery.py:15`의 단일 상수 `PRE_SEPARATION_SILENCE_RMS`에 있다.

### VAD

- `silero_vad.get_speech_timestamps`를 사용한다.
- 현재 기본 실시간 실행에서는 `--vad`로 활성화하고, latest 60초 결과는 200 ms minimum speech duration을 사용했다.
- speaker별 분리 output에 대해 speech duration, speech ratio, RMS, peak를 기록한다.
- speech가 없으면 해당 speaker STT를 skip하지만 separated window 자체와 subtitle state 처리는 유지한다.

### STT

- 현재 Live/최신 저장 결과의 실사용 backend는 FunASR `SenseVoiceSmall`이며 `device="cuda"`, `language="ko"`, `use_itn=True`를 사용한다.
- `phase3/run_realtime_pipeline.py`의 `transcribe_audio`는 SenseVoice에 메모리 NumPy array를 직접 전달한다.
- faster-whisper backend도 아직 존재한다. `faster-whisper base`, CUDA, `float16`, `language="ko"`, `beam_size=5`, `condition_on_previous_text=False` 설정을 사용한다.
- `phase3/benchmark_stt.py`는 두 모델을 동일한 diagnostic WAV 전체에 대해 비교하는 독립 benchmark다.
- SenseVoice 결과에는 `<|ko|>`, `<|Speech|>` 같은 special token이 반환될 수 있으며 실시간 경로에서는 tag 제거 정규식을 적용한다. 독립 benchmark 출력에는 raw special token이 남은 결과가 저장되어 있다.

### Subtitle

- `SubtitleAssembler`가 인접 overlap window의 suffix/prefix를 raw/token/normalized/fuzzy 방식으로 비교해 중복 문자를 제거한다.
- fuzzy threshold는 `0.88`, 최소 문자 수와 overlap 판단 규칙은 `phase3/subtitle_assembler.py`에 있다.
- `SpeakerSubtitleState`는 speaker별 `utterance_id`, stable text, tentative text, hypothesis history를 관리한다.
- history size는 3이다.
- 각 STT 결과는 PARTIAL 이벤트로 반영되고, VAD silence 또는 pipeline end에서 FINAL 이벤트를 만든다.
- 기본 silence finalize 기준은 2000 ms다.
- consensus는 반복 관측된 prefix를 stable로 승격하고, tentative hypothesis correction/tail correction을 처리한다.

### Frontend/WebSocket

- `SubtitleWebSocketServer`는 Python 별도 thread에서 asyncio WebSocket server를 실행한다.
- endpoint 기본값은 `ws://127.0.0.1:8765`이다.
- event queue는 기본 maxsize 32이고 가득 차면 오래된 이벤트를 버려 최신 자막을 유지한다.
- `phase4/web/app.js`는 speaker별 최신 4개 이벤트를 표시하고 같은 `utterance_id`의 partial/final을 갱신한다.
- browser는 receipt event를 다시 보내 브라우저 수신 latency를 기록한다.

## 4. Phase별 개발 과정

### Phase 0: 오프라인 기본 검증

- 목표: 두 개의 한국어 WAV를 같은 시작 시점에 합성하고, MossFormer2로 두 speaker를 분리한 뒤 STT까지 연결한다.
- 구현: `phase0/mix.py`가 mono/16 kHz 변환과 resample 후 peak를 0.98로 제한해 `mixed.wav`를 만든다. `phase0/separate.py`가 ClearVoice `MossFormer2_SS_16K`를 로드하고 두 output WAV를 저장한다. `phase0/transcribe.py`가 faster-whisper base로 각 WAV를 전사한다.
- 결과: `phase0/output`에 mixed, speaker WAV, txt 결과가 존재한다. 이 단계는 전체 모델/데이터 계약을 확인한 offline baseline이다.

### Phase 1: WASAPI loopback 및 오디오 품질 진단

- 목표: 실제 Windows system audio를 loopback으로 녹음하고, sample rate/channel 변환이 분리 품질을 훼손하는지 확인한다.
- 구현: `phase1/capture.py`가 48 kHz capture, mono 변환, 16 kHz resample을 수행한다. `phase1/diagnostic`에는 8-channel capture, 직접 mixed signal, channel별 signal, playback/capture offset 확인 코드가 있다.
- 확인된 구조: 현재 실시간 pipeline은 multi-channel 평균이 아니라 WASAPI 첫 채널 `raw[:, 0]`을 사용한다.
- 결과: loopback 및 channel별 diagnostic WAV가 저장되어 있다. 다채널 downmix 품질 문제는 진단되었지만, 모든 시스템/장치에서 최적이라고 확정된 상태는 아니다.

### Phase 2: MossFormer2 성능 및 안정성

- 목표: RTX 3080 10 GB에서 5초 separation이 실시간보다 빠르게 안정적으로 동작하는지 확인한다.
- 구현: `phase2/benchmark_cuda_amp.py`에 CUDA synchronize, inference mode, AMP FP16, VRAM 측정이 있다. `benchmark_gpu_stability.py`는 warm-up 후 반복 측정한다. `run_continuous.py`, `run_short_chunks.py`는 1/2/5초 chunk를 비교한다.
- 문제: 같은 환경에서도 CUDA 초기화/warm-up/측정 경로 차이로 1.8초 수준과 수십 초 수준이 모두 관찰되었다.
- 해결/결과: 모델 1회 로드, warm-up, 전후 synchronize, 반복 측정 방식으로 안정화했고, Phase 2-F 기준 5초 RTF < 1 및 반복 중 VRAM 증가 없음이 확인되었다. 이후 통합 pipeline에는 AMP와 inference mode가 유지됐다.

### Phase 3-B: Capture -> Separation -> STT 통합

- 목표: 5초 chunk 30초 입력을 capture부터 두 speaker STT까지 하나의 bounded queue pipeline으로 연결한다.
- 구현: capture worker, audio queue, separation worker, separated queue, STT worker를 만들고 모델을 startup에 한 번만 로드한다. memory NumPy array를 STT에 직접 전달한다.
- 결과: 저장된 Phase 3 계열 결과에서 30초 5초 chunk 테스트가 capture/separation/STT 6/6 및 12/12로 완료되었다. 초기 사용자 체감 지연은 5초 window 자체가 만든 약 5초 대기였다.

### Phase 3-C: chunk latency 비교

- 목표: 1/2/3/5초 chunk를 같은 30초 실시간 pipeline에서 비교한다.
- 구현: `phase3/benchmark_chunk_latency.py`와 `phase3/output/chunk_benchmark` 결과가 있다.
- 결과: 1초는 separation 평균 0.368초지만 2 drops와 separation success 28/30이었다. 2/3/5초 결과는 저장 JSON을 기준으로 비교할 수 있다. 이 실험에서 단순히 가장 짧은 chunk를 선택하지 않고 queue drop, RTF, transcript fragment/empty/repeat를 함께 보도록 했다.
- 후속 방향: overlap 기반 3초 window/2초 stride가 STT 품질과 지연의 타협점으로 채택됐다.

### Phase 3-D/E: overlap 및 incremental subtitle

- 목표: 3초 window/2초 stride의 중복 오디오에서 중복 transcript를 제거하고, 실시간 자막을 incremental하게 갱신한다.
- 구현: `run_overlap_pipeline.py`의 `overlap_candidate`, `analyze_overlap_duplicates` 및 `subtitle_assembler.py`의 raw/token/normalized/fuzzy suffix-prefix matching.
- 결과: 저장 결과에서 overlap duplicate 분석과 assembled transcript가 확인된다. 중복 제거는 완전한 문장 품질 보장이 아니라 window 경계 중복을 줄이는 처리다.

### Phase 4-A: WebSocket 및 UI MVP

- 목표: Python subtitle event를 브라우저로 보내고 speaker별 자막을 표시한다.
- 구현: `phase4/subtitle_websocket.py`, `phase4/web/index.html`, `app.js`, `styles.css`, `phase4/run_websocket_demo.py`, `phase4/test_websocket_mvp.py`.
- 결과: `phase4_a_e2e.json`에는 WebSocket 포함 30초 실행 성공 기록이 있고, client log/json도 존재한다. UI는 로컬 MVP이며 서버 인증/배포/React/Next.js는 구현되지 않았다.

### Phase 4-B: diagnostic WAV

- 목표: 실제 loopback 원본과 speaker separation output을 WAV로 저장해 원본/STT와 separated/STT 품질을 분리 진단한다.
- 구현: `--diagnostic-audio`가 original 연속 stream과 speaker output을 저장한다. overlap window를 단순 stitch해 중복 오디오를 피한다. 종료 후 original/speaker_0/speaker_1에 별도 STT를 수행한다.
- 결과: `phase3/output/diagnostic/*.wav` 세 파일이 존재하며 모두 16 kHz mono PCM WAV다.

### Phase 4-C: faster-whisper와 SenseVoiceSmall 비교

- 목표: 동일한 전체 WAV에 두 STT 모델을 적용해 입력 품질과 모델 차이를 분리한다.
- 구현: `phase3/benchmark_stt.py`가 faster-whisper base CUDA float16과 FunASR SenseVoiceSmall CUDA를 순차 실행하고 duration/RMS/peak/time/RTF/GPU memory/transcript를 JSON에 저장한다.
- 결과: 원본에서 두 모델 모두 장문 transcript를 생성했고, speaker_1 artifact 파일에서도 두 모델의 짧은 오인식이 발생했다. 이 결과만으로 separation artifact와 STT 자체 문제의 비중을 확정하지는 않았다.

### Phase 4-D~H: SenseVoice/VAD/assembler/state

- 저장된 `live_sensevoice_30s.json`, `live_sensevoice_vad_30s.json`, `two_speaker_assembler_30s.json`, `two_speaker_subtitle_state_30s.json`에서 SenseVoice backend, VAD, assembler, subtitle state 단계의 실행 결과가 확인된다.
- VAD 적용 시 speech 없는 speaker의 STT 호출을 건너뛰고 queue/backlog를 줄이는 효과가 확인된다.
- 상태 관리에는 PARTIAL/FINAL, utterance id, silence finalization, tentative/stable 및 consensus history가 추가됐다.

### Phase 4-I: consensus, numerical recovery, silence gate

- 목표: AMP non-finite와 완전 무음 입력을 pipeline failure로 만들지 않고, 기존 downstream 상태 흐름을 유지한다.
- 구현: `separation_recovery.py`의 finite input 검사, AMP non-finite 검출, FP32 retry, `PRE_SEPARATION_SILENCE_RMS = 1e-5` gate. `run_overlap_pipeline.py`가 gate output을 VAD/queue로 전달하고 recovery metrics를 JSON에 기록한다.
- 회귀 테스트: `test_separation_recovery.py`, `test_pre_separation_silence_gate.py`, subtitle 관련 세 regression이 존재한다.
- 최신 결과: `live_consensus_60s_silence_gate.json`은 60초, capture 29/29, separation 29/29, STT 29/29 windows, speaker inputs 58, drop 0, error 0, silence gate 3회, AMP non-finite 0, FP32 fallback 0, peak 2996 MiB로 저장되어 있다.

## 5. 현재 실시간 파이프라인

실제 `run_overlap_pipeline.py` 기준 흐름은 다음과 같다.

```text
WASAPI loopback 48 kHz
  -> raw[:, 0] 첫 채널 선택
  -> resample 16 kHz mono float32
  -> retained tail + 새 stride로 3초 window 생성
  -> bounded audio queue(maxsize=2)
  -> finite input 검사
       ├─ NaN/Inf: invalid input 기록, 해당 window skip
       └─ finite
            -> pre-separation silence gate
                 ├─ RMS <= 1e-5: zero speaker_0/1 output 생성
                 └─ non-silence: MossFormer2 AMP FP16
                                  └─ non-finite output: FP32 fallback
  -> bounded separated queue(maxsize=2)
  -> speaker_0 / speaker_1
  -> Silero VAD (--vad)
       ├─ no speech: STT skip
       └─ speech: SenseVoiceSmall 또는 faster-whisper
  -> overlap assembler
  -> SpeakerSubtitleState consensus
  -> PARTIAL / FINAL subtitle event
  -> bounded WebSocket event queue
  -> browser UI
```

무음 gate는 separation을 건너뛰지만 separated queue, VAD, subtitle state를 우회하지 않는다. 따라서 무음 output의 시간축과 downstream window count가 유지된다.

## 6. 현재 성능

### 최신 Live 결과

근거: `phase3/output/live_consensus_60s_silence_gate.json`

| 항목 | 실제 값 |
|---|---:|
| 테스트 | 60초, 3초 window / 2초 stride |
| Capture | 29/29 |
| Separation | 29/29 |
| STT windows | 29/29 |
| Speaker STT inputs | 58 |
| Drop | 0 |
| Errors | 0 |
| Separation 평균/중앙/p95/최대 | 0.300 / 0.281 / 0.364 / 1.159초 |
| Window STT 평균/중앙/p95/최대 | 0.131 / 0.122 / 0.184 / 0.189초 |
| Post-capture latency 평균/중앙/p95/최대 | 0.436 / 0.447 / 0.532 / 1.288초 |
| 초기 subtitle latency | 3.097초 |
| Window 시작 기준 평균 latency | 3.436초 |
| Audio queue depth/backlog 최대 | 1 / 0 |
| Separated queue depth/backlog 최대 | 1 / 0 |
| VAD STT skip | 23/58 |
| Silence gate | 3/29 windows, ratio 0.1034 |
| AMP attempts/non-finite | 26/0 |
| FP32 fallback | 0회 |
| GPU after both models / overall peak | 2695 / 2996 MiB |
| CUDA OOM | 없음 |
| WebSocket dropped_oldest | 0 |

이 결과는 현재 저장된 가장 최근의 성공적인 Live 결과다. 다만 transcript quality는 별도 성공 조건이 아니며, 저장 결과의 speaker_1 final segment에는 짧고 잘못된 텍스트가 다수 남아 있다.

### Phase 4-C diagnostic STT 비교

근거: `phase3/output/stt_benchmark.json`

| 입력 | 모델 | 처리시간 | RTF | 저장 transcript 관찰 |
|---|---|---:|---:|---|
| original.wav, 30초 | faster-whisper base | 5.102초 | 0.170 | 긴 한국어 문장 생성, 오인식 일부 존재 |
| original.wav, 30초 | SenseVoiceSmall | 0.480초 | 0.016 | 긴 문장 생성, special token 포함 |
| speaker_0.wav, 29초 | faster-whisper base | 1.852초 | 0.064 | 원본과 유사한 긴 문장 |
| speaker_0.wav, 29초 | SenseVoiceSmall | 0.118초 | 0.004 | 긴 문장 생성, 오인식 존재 |
| speaker_1.wav, 29초 | faster-whisper base | 10.819초 | 0.373 | “이걸로” 한 문장 |
| speaker_1.wav, 29초 | SenseVoiceSmall | 0.098초 | 0.003 | “당일 클 이 할.” 수준의 artifact 오인식 |

이 결과는 speaker_1에 speech가 거의 없거나 artifact가 섞인 상황에서도 두 모델이 비어 있지 않은 텍스트를 만들 수 있음을 보여준다. 따라서 현재 STT 오인식 원인은 아직 하나로 확정되지 않았다.

## 7. 주요 문제와 해결 상태

| 문제 | 원인/관찰 | 해결 방법 | 현재 상태 |
|---|---|---|---|
| WASAPI multi-channel 품질 | 초기 diagnostic에서 장치 채널별 신호와 downmix 차이가 문제 후보로 확인됨 | 현재 pipeline은 `raw[:, 0]`을 사용 | 부분 해결. 장치별 최적성 미검증 |
| GPU separation latency 변동 | CUDA 초기화, warm-up, synchronize, 측정 경로 차이 | 1회 모델 로드, warm-up, AMP, 전후 synchronize, 반복 benchmark | 해결됨으로 볼 수 있으나 환경 의존성은 남음 |
| GPU VRAM/OOM | separation+STT 동시 로딩과 장시간 실행 위험 | CUDA 측정, 모델 1회 로드, queue 제한 | 최신 60초 결과 OOM 없음, 장시간 미검증 |
| 1/2초 chunk queue pressure | 짧은 chunk에서 상대적 STT/separation 변동과 drop 발생 | 3초 window/2초 stride overlap 비교 | 1초 조건은 30개 중 2 drop, 3/2 구조 채택 |
| AMP NaN/Inf | separation output이 non-finite가 될 수 있음 | FP32 fallback 및 window 단위 failure isolation | 해결됨. 최신 60초 non-finite 0 |
| 완전 무음에서 MossFormer2 불안정 | silence 입력에서 모델 호출 자체가 불필요하고 NaN 위험 | pre-separation silence gate, zero speaker output | 해결됨. 최신 gate 3회 정상 작동 |
| STT 지연 | faster-whisper 또는 artifact 입력에서 호출 시간이 길어질 수 있음 | SenseVoice와 VAD 도입, VAD skip | 부분 해결. STT 품질은 미해결 |
| VAD | 분리된 무음 channel도 STT에 들어갈 수 있음 | Silero VAD로 no-speech STT skip | 동작 확인, false speech artifact는 남음 |
| STT 오인식 | 원본/분리 output과 모델별 품질 차이가 존재 | Phase 4-C diagnostic WAV 비교까지 수행 | 미해결, 원인 분리 필요 |
| overlap 중복 | 3초 window가 1초 overlap을 가짐 | assembler suffix/prefix/fuzzy 제거 | 부분 해결. 오인식이 있으면 중복/삭제 판단도 흔들림 |
| consensus instability | window별 STT hypothesis가 달라짐 | stable/tentative, history size 3, tail correction | 동작하지만 품질 입력에 의존 |
| WebSocket backlog | 이벤트가 빠르게 쌓일 가능성 | bounded event queue, oldest drop 정책 | 최신 결과 drop 0, 장기/느린 browser 미검증 |
| UI | MVP 표시와 reconnect 필요 | local HTML/JS, utterance 갱신, receipt latency | 동작 확인, 제품 UI/배포 미완성 |

## 8. 현재 남아 있는 핵심 문제

### 최우선: STT 및 separation artifact 원인 미확정

현재 한 명이 말하는 영상에서도 speaker_1에서 VAD speech가 검출되거나 짧은 잘못된 transcript가 생성된다. `live_consensus_60s_silence_gate.json`에는 58 speaker inputs 중 35 active, 23 VAD skip이 기록되어 있고, speaker_1 final segments에는 “이건 클릭리은 몇졌잖아.”, “패키지.”, “그.” 같은 짧은 결과가 남아 있다.

현재 가능한 원인은 다음과 같이 분리해서 봐야 한다.

- A. WASAPI 원본 음질: 현재 첫 채널만 사용하므로 channel selection/downmix 문제가 남아 있다.
- B. MossFormer2 artifact: Phase 4-C에서 speaker_1 WAV peak가 1.0이고 RMS가 0.041인 반면 transcript는 매우 짧아 artifact 가능성이 있다.
- C. SenseVoice 자체 인식: original과 speaker_0도 자연스러운 문장과 오인식이 섞여 있어 모델 품질 영향도 있다.
- D. 3초 window: window 경계와 overlap이 문장 절단 및 반복을 만든다.
- E. pipeline 영향: VAD false positive, assembler/consensus가 잘못된 hypothesis를 표시할 수 있다.

현재 결과만으로 A~E 중 하나를 해결됨으로 판정하면 안 된다. 특히 `phase3/output/stt_benchmark.json`은 원본, speaker_0, speaker_1에 대한 offline 비교 자료이지만, 같은 Live window와 같은 시점의 원본-vs-separated 대조를 자동화한 결과는 아니다.

### 그 외 남은 문제

- 장기 Live 실행에서 GPU memory, WDDM, device 상태가 60초 이후에도 계속 안정적인지는 미검증이다.
- WASAPI 첫 채널 선택이 모든 출력 장치에서 충분한지 미검증이다.
- speaker_1의 false VAD와 artifact STT를 줄이는 방법은 아직 구현하지 않았다.
- subtitle 품질 평가는 reference transcript가 없어 WER/CER 정량화가 불가능하다.
- UI는 로컬 MVP이고 authentication, reconnect state replay, deployment, browser compatibility는 제품 수준으로 검증되지 않았다.
- 실제 diarization/화자 이름/장기 speaker identity는 구현되지 않았다.

## 9. 다음 세션의 최우선 작업

다음 작업은 수정 전에 같은 Live 오디오를 기준으로 원본과 분리 output을 직접 비교하는 진단이다.

```text
동일한 Live capture
  -> original audio -> SenseVoice
  -> MossFormer2 speaker_0 -> SenseVoice
  -> MossFormer2 speaker_1 -> SenseVoice
```

필요한 조사/수정 지점:

1. `phase3/run_overlap_pipeline.py`의 capture worker에서 원본 continuous stream과 각 window의 capture timestamp를 보존한다.
2. separation worker의 `speaker_0/1` output 및 `pre_separation_silence`/speaker RMS/peak를 함께 보존한다.
3. 가능하면 같은 window 또는 같은 stitched 30~60초 구간에 대해 original, speaker_0, speaker_1을 동일 SenseVoice 설정으로 offline 전사한다.
4. transcript뿐 아니라 RMS, peak, VAD timestamps, STT time, RTF를 나란히 기록한다.
5. original도 이미 잘못되면 capture/SenseVoice를 우선 의심하고, original은 정상인데 speaker_1만 무너지면 separation artifact를 우선 의심한다.

이번 진단 전에는 MossFormer2 교체, SenseVoice 교체, window/stride 변경, VAD threshold 대폭 변경, consensus 대규모 수정, WebSocket 구조 변경을 임의로 하지 않는 것이 안전하다.

## 10. 현재 상태 한 줄 평가

| 영역 | 상태 | 근거 |
|---|---|---|
| Audio Capture | 동작하지만 개선 필요 | WASAPI Live 60초 capture 29/29, 현재는 첫 채널 선택 |
| Speech Separation | 동작하지만 개선 필요 | MossFormer2 CUDA/AMP, gate/recovery 정상; artifact 품질은 미확정 |
| Realtime Performance | 완료에 가까움 | 최신 60초 queue backlog/drop 0, separation RTF는 충분히 낮음 |
| VAD | 동작하지만 개선 필요 | 58개 speaker input 중 23개 STT skip, false active 사례 존재 |
| STT | 동작하지만 개선 필요 | SenseVoice 정상 실행이나 한국어 오인식과 artifact transcript 존재 |
| Subtitle Stabilization | 동작하지만 개선 필요 | PARTIAL/FINAL/consensus 구현, 입력 hypothesis 품질에 의존 |
| WebSocket | 완료에 가까움 | 최신 결과 published/broadcast 43, dropped_oldest 0 |
| Frontend | 동작하지만 개선 필요 | 로컬 HTML/JS MVP, 제품 UI/배포 미완성 |
| Overall | 동작하지만 개선 필요 | 전체 파이프라인은 연결되고 안정성 PASS이나 STT/분리 품질 진단이 남음 |

</details>
