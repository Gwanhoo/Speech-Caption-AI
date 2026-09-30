2026-09-30 정확도 pipeline audit 및 구현 결과

1. 진단한 핵심 원인

실제 remote GUI 경로는 `phase5/live_caption_controller.py` → `phase3/run_overlap_pipeline.py`의 Windows capture/queue → `phase4/remote_gpu_client.py` → GPU server의 MossFormer2/Silero/faster-whisper → Windows speaker tracker/assembler/state → WebSocket 및 Qt structured event이다. GUI는 `event.text`를 같은 utterance의 PARTIAL로 교체한다. UI가 원문 중복을 만드는 것이 아니라 이미 조립된 텍스트를 받는다.

- RAW STT 오류와 assembler의 수정 불가능한 append 동작이 함께 존재했다.
- 기존 fuzzy는 최소 6글자, 유사도 0.88이고, 매칭되어도 새 prefix를 버려 이전 오인식을 남겼다. `안뇽하세요`처럼 5글자인 재인식은 빠졌다.
- state에 전달하는 전체 hypothesis는 이전 텍스트를 복사한 값인데, 단순 extension을 새 관측의 지지로 취급했다. history 3개를 보유하지만 현재 실행의 안정화 판단은 실제로 직전 hypothesis 비교였다.
- 단일 음성을 MossFormer2에 넣어도 잔여 출력이 VAD를 통과하고 STT 환각/중복을 만들었다. 큰 모델에서도 재현됐다.
- 첫 요청의 CUDA/decoder 초기화와 3초 수집 자체가 첫 자막 지연에 기여했다.

2. 원인별 증거

| 단계 | 코드/실측 근거 | 판단 |
|---|---|---|
| Capture | WASAPI 48kHz, `raw[:, 0]`, `resample_poly` 16kHz, 이후 3초/2초 | 오른쪽 채널 단독 발화가 사라질 수 있다. 실제 Windows 장치 영향은 미검증 |
| Resampling | B.wav를 48kHz로 만든 대체 입력에서 연속 처리와 3+2초 블록 처리 비교: 240,000 samples 동일, 상대 RMS 차이 -54.51dB, 최대 차이 0.0529, 차이 >1e-6인 samples 114개 | 블록 경계 영향은 존재하나 이 자료로 주요 오인식 원인이라고 볼 수 없음 |
| Transport | 요청 FLOAT WAV, 응답 f32 binary/base64 | live 전송에서 ±1 초과 값을 clipping하지 않음 |
| Diagnostic | 기존 `--save-wav`/연속 WAV가 PCM16. 이번 separated peak 최대 1.224, ±1 이상 samples 총 16개 | offline 비교 자료가 live STT 입력과 달라질 수 있었음. FLOAT로 수정 |
| Separation | ClearVoice 0.1.2 `utils/decode.py`가 각 output RMS를 입력 RMS로 다시 맞춤 | 출력 RMS 비율로 약한 화자/잔여 음성을 판단할 수 없음 |
| 단일 A | window 0의 주 출력과 원본 상관계수 0.999969, 잔여 출력 VAD 46.5%. base `오늘의 영상은`, small `구독과 좋아요 부탁드려요!` 등 생성 | 원본이 거의 그대로 주 출력에 있어도 잔여 환각 발생 |
| 단일 A | window 5의 두 output과 원본 상관계수 0.857/0.610, 두 출력 모두 VAD 통과 | 하나의 원본 화자가 두 output에 나뉘는 사례. lexical secondary 일괄 삭제는 위험 |
| 혼합 A+B | 7 windows 모두 두 output VAD active, 두 원본 내용이 각각 전사됨 | separation 전체 bypass를 기본값으로 선택할 근거가 없음 |
| VAD | 전체 window를 STT에 그대로 전달; timestamps로 자르지 않음. 200→100ms 실험에서 주 음성 검출 동일, A 잔여 active만 2→3 | 기본 200ms 유지. 아주 짧은 발화 전체 skip 가능성은 남음 |
| RAW STT | base B window 1: `10분 초 공원으로`; small: `집 근처 공원으로` | 모델 오인식과 조립 문제를 별개로 관찰 |
| 조립 | small 실측 기존 결과 `간단한 하게`; 수정 후 `간단하게`. `공원에는 운동을 하거든요. 공원에는 운동을 하거나…`도 교체 | 같은 RAW 입력에서 수정 효과 재현 |
| Silence/noise | digital silence 2 windows에서 전 모델 출력 0. 고정 seed Gaussian noise는 분리 후 1/2 windows VAD 양성 | 일반 무음과 separation artifact는 별도 문제 |

현재 설치된 faster-whisper 1.2.1의 실행 옵션은 `ko`, beam 5, `condition_on_previous_text=False`, `initial_prompt=None`, word timestamps off, 내부 VAD off이다. 명시하지 않은 temperature는 **0 고정이 아니라 `[0, .2, .4, .6, .8, 1]` fallback**이며 no-speech .6, log probability -1, compression ratio 2.4가 기본값이다. 이번에는 이 decoding 설정을 바꾸지 않았다. `available_models()`와 실제 signature로 모델/API를 확인했다. [공식 faster-whisper 문서](https://github.com/SYSTRAN/faster-whisper), [공식 release 기록](https://github.com/SYSTRAN/faster-whisper/releases), [MossFormer2 모델](https://huggingface.co/alibabasglab/MossFormer2_SS_16K)을 참고했다.

기존 `benchmark_stt_context.py`의 5초/prompt A–D 실험과 `benchmark_stt_model_size.py`의 base pseudo-reference를 확인했다. 과거 75초 diagnostic와 해당 결과 JSON은 이 checkout에 없다. 그 결과를 새 실측으로 주장하거나 5초/prompt 실험을 반복하지 않았다. 현재 자료는 저장소 A/B fixture 및 phase4의 기존 A5000 로그다.

3. 수정한 파일

- `phase3/subtitle_assembler.py`
- `phase3/run_overlap_pipeline.py`
- `phase3/secondary_leakage_diagnostics.py`
- `phase3/audio_diagnostics.py` (신규)
- `phase4/gpu_processing_server.py`
- `phase3/benchmark_accuracy.py` (신규)
- `phase4/benchmark_caption_latency.py` (신규)
- `phase3/test_accuracy_regressions.py` (신규)
- `phase4/test_gpu_processing_server.py`
- `phase3/accuracy_audit_summary.json` (신규 측정 요약)
- `README.md`, `phase4/REMOTE_GPU.md`, 이 문서

4. 각 수정 내용

- 인접 window의 최대 48글자 suffix/prefix를 음절 및 분해한 한글 자모로 비교한다. 짧은 매칭은 시작 글자, 어절 경계, 음절 일치도도 요구한다. 확인된 seam을 현재 hypothesis로 **교체**하고 나머지만 추가한다. 전체 session을 fuzzy 검색하지 않는다.
- exact seam도 새 문자열에서 재조립하여 `아침 에`, `간단 하게` 같은 어절 내부 공백을 막는다. gap 뒤에는 overlap을 적용하지 않고 실제 반복을 보존한다. FINAL 이후 session history는 수정하지 않는다.
- `confirmed_prefix_length`는 원본 window끼리 exact하게 지지한 연속 prefix만 기록한다. production state에 이 근거를 전달해 복사된 텍스트의 가짜 consensus를 막는다. FINAL 규칙은 VAD silence/pipeline end를 유지한다. 독립 지지가 부족한 긴 prefix는 FINAL까지 tentative일 수 있다.
- 두 출력이 같은 window에서 파형 상관 절댓값 ≥0.98이고 정규화 전사문도 완전히 같은 4글자 이상일 때만 secondary 중복을 제거한다. 텍스트 유사도나 작은 RMS만으로 억제하지 않는다. 이 좁은 조건은 unit regression으로 검증됐으며 이번 실제 잔여 환각에는 적용되지 않았다.
- channel diagnostics와 `--capture-channel first|mean`을 추가했다. 기본 first를 유지한다. channel별 RMS/peak/포화 sample 수, 오른쪽만 활성, downmix 상쇄 후보를 JSON에 기록한다. 실제 장치 evidence 없이 자동 downmix를 바꾸지 않았다.
- remote server 기본 STT를 small로 선택하고 `--whisper-model base|small|medium|large-v3-turbo|large-v3`를 제공한다. 선택 모델을 health/response에 기록하며 base cache를 다른 모델로 잘못 사용하는 것을 방지한다.
- 요청을 처리하는 동일한 지속 worker thread에서 startup warm-up을 완료한 다음 ready가 된다. `--no-warmup`으로 cold-start benchmark를 재현할 수 있다.
- HTTP schema version 1, 두 raw slots, waveform, 기존 subtitle event 필수 필드, Windows/서버 책임 경계 및 Phase 5 UI/overlay를 유지했다.

변경 우선순위는 재현 가능한 assembler 오류(효과 큼, GPU 비용 없음) → 진단 무손실 저장/채널 계측(위험 낮음) → 모델 5개 비교(다운로드 및 GPU 비용) → 선택 모델/warm-up(실측 후 적용)이다. 자동 single-speaker bypass, 에너지 기반 lexical 삭제, 5초/prompt 재도입, word timestamp 기반 전면 재작성은 충분한 검증 근거가 없어 적용하지 않았다.

5. 수행한 benchmark

- A.wav/B.wav 각각 첫 15초, 같은 구간의 0.5×(A+B), digital silence 5초, seed 42의 Gaussian noise 5초(RMS 약 .002).
- 3초 window/2초 stride: 25 windows, original/두 separated input 75개. VAD 후 **모델당 52회, 5개 모델 총 260회 STT**. separation은 한 번 저장하여 모든 모델에 정확히 같은 FLOAT 입력 제공.
- RAW/STT confidence 및 시간, 원본/분리 음량, VAD timestamps, source correlation, tracker mapping, before/after hypothesis/event, adjacent repeated token heuristic, CER/WER 및 insertion/deletion 계산 경로, nvidia-smi process VRAM 기록.
- 사람이 검수한 reference가 없으면 speech CER/WER/누락률은 `null`. 기존 base 전사문을 정답으로 사용하지 않음. insertion/deletion은 reference edit count이며 의미적 hallucination 검출기가 아님.
- assembler 텍스트 fixture 8개: 오타 재인식, 어미 교정, 짧은 seam, 진짜 반복/다른 문장 보존. 실제 음성 CER과 분리해서 보고.
- VAD 100/200ms, 블록 resampling 대조.
- paced WAV → queue(max 3, full이면 새 window drop) → 실제 localhost HTTP GPU → tracker/assembler/state → 실제 WebSocket 수신. base cold / small cold / small warm 각각 27 requests, 36 subtitle events, 총 81 requests. WASAPI/외부 proxy/Qt paint는 제외.
- 최종 테스트: **74 passed, 21 subtests**, 기존 main 방식 regression **13개 PASS**. Linux에서 soundcard를 명시적으로 mock하여 수집한 결과이며 WASAPI 성공을 의미하지 않는다.

6. Before / After 수치

| 모델 | 두 output STT 평균 / p95 (초) | 저장된 separation+VAD+STT 평균 / p95 (초) | STT process peak MiB | noise 잘못된 출력 글자 수 |
|---|---:|---:|---:|---:|
| base | .385 / 1.312 | .778 / 1.607 | 542 | 11 |
| small | .396 / .692 | .789 / 1.387 | 958 | 0 |
| medium | 1.071 / 4.514 | 1.464 / 4.810 | 2974 | 11 |
| large-v3-turbo | .443 / 1.022 | .836 / 1.580 | 2366 | 5 |
| large-v3 | .946 / 1.775 | 1.339 / 3.450 | 4318 | 5 |

시간 표는 **음성 21 windows** 기준, noise 글자 수는 **음성이 없는 2 windows** 기준이다. 모델별 단일 실행이며 신뢰구간은 없다. 공통 separation/VAD를 저장해 합친 offline 합계이므로 HTTP 지연이 아니다. VRAM은 CTranslate2를 포함한 해당 STT process 측정이며 separator 동시 로딩 peak가 아니다. 실제 small 서버는 검증 종료 시 process 1430 MiB였다.

| 조립 텍스트 fixture 8개 / 137 정답 글자 | 이전 | 이후 |
|---|---:|---:|
| 예상 display와 불일치하는 사례 | 4 | 0 |
| CER (공백/문장부호 제외) | 16.06% | 0% |
| 삽입 / 삭제 / 치환 | 22 / 0 / 0 | 0 / 0 / 0 |

이 표는 **구성한 조립 회귀 fixture**의 성적이다. 실제 발화 전체의 CER이 0%라는 뜻이 아니다. `안뇽하세요 → 안녕하세여…`에서는 새 hypothesis로 바꿀 뿐 정답 `안녕하세요`를 만들어 내지 않는다. 세 번째 관측이 `안녕하세요…`일 때 추가 교정이 가능한 것도 검증했다.

7. 선택한 STT 모델 및 선택 이유

새 remote server 기본값은 **small / CUDA float16**이다. base보다 이 샘플에서 `집 근처`, `선선해서` 등의 출력이 나았고 STT 평균 추가 비용은 약 11ms/window였다. 고정 noise에서는 small만 잘못된 자막이 없었다. turbo도 실시간 후보지만 작은 표본에서 small보다 더 정확하다는 근거는 없었고 메모리가 더 컸다. medium/large는 긴 지연 tail이 있었다. 이는 제한된 샘플의 운영 선택이며 한국어 고유명사·게임 문맥을 포함한 일반 정확도 우위는 **미검증**이다.

8. 정확도 개선 정도

정량 확인: 구성한 조립 fixture의 중복/오타 잔존 제거, synthetic nonspeech의 base 11글자 → small 0글자. 실제 RAW 재생에서도 `간단한 하게` 교정 및 `공원에는…` seam 중복 감소를 확인했다. 검수된 speech reference가 없으므로 **실제 한국어 전체 CER/WER, 누락률, 고유명사 정확도의 개선율은 미검증**이다. 모든 original/분리 전사문과 잔여 output을 결과에 보존했다.

9. latency 변화

| 실제 localhost paced 경로 | base, cold | small, cold | small, warm-up |
|---|---:|---:|---:|
| 첫 단일 A 자막 (3초 수집 포함) | 7.010s | 5.677s | 3.878s |
| capture-ready → WebSocket 수신 평균 | .940s | .867s | .752s |
| 동일 수신 지연 p95 | 3.096s | 1.874s | 1.012s |
| queue wait 최대 | 2.007s | .676s | .0014s |
| queue depth 최대 | 2 | 1 | 1 |
| drops / requests | 0 / 27 | 0 / 27 | 0 / 27 |

warm-up은 서버 준비에 **2.660초**를 더 쓴다. 기다림을 요청 전에 수행하는 것이다. 수정 후 small assembler p95는 저장된 RAW replay에서 약 **0.736ms/호출**. 초기 수집 3초와 갱신 간격 2초는 그대로이며, Windows 네트워크/장치와 화면 표시까지의 지연은 **미검증**이다. 이벤트 기준 분포라서 요청 roundtrip 분포와 같지 않다.

10. 아직 남은 문제

- small에서도 단일 A 잔여 output의 환각과 같은 화자가 두 output으로 갈라지는 문제가 남는다. 새 중복 억제는 이를 모두 해결하지 않는다.
- 서로 크게 다르게 인식된 seam, 짧은 의미 차이, timestamp 없는 반복/교정 판별은 불완전하다. 새 hypothesis도 틀릴 수 있다.
- 무음이 없는 긴 발화에는 utterance 길이 제한/문장별 finalization을 도입하지 않았다. 긴 PARTIAL의 가독성과 memory/session 길이 증가가 남는다.
- VAD 200ms 미만의 짧은 독립 발화, 음소가 window 경계에서 잘리는 문제, 장치 discontinuity는 별도 corpus가 필요하다.
- word timestamps는 off이며, 장시간/음악/게임 효과음/조용한 실제 두 번째 발화/다채널 WASAPI에서는 미검증이다.
- 이 작업에서 원래 실행 중이던 서버를 재시작하지 않았다. 새 기본값 적용에는 서버 재시작 및 Windows에 수정 코드 반영이 필요하다. 검증용 서버만 별도 포트에서 띄웠다.

11. Windows Live Test에서 직접 확인할 것

RunPod 서버를 새 코드로 시작한다. base 대조는 마지막 옵션만 `base`로 바꾼다.

```bash
python phase4/gpu_processing_server.py --host 0.0.0.0 --port 8787 --whisper-model small
```

Windows에서 동일 음원을 재생하고 원본/분리 WAV 및 모든 timing을 남긴다.

```powershell
python phase3/run_overlap_pipeline.py --live --duration 75 `
  --processing-mode remote --remote-server-url "$env:GPU_SERVER_URL" `
  --assemble --websocket --save-wav --latency-diagnostics `
  --json-output phase3/output/windows_accuracy_small.json
```

`phase3/output/overlap_debug/window_NNN_mixed.wav`, `raw_slot_N.wav`, `logical_speaker_N.wav`가 같은 window의 FLOAT 비교 자료다. 진단 폴더는 다음 실행에서 같은 이름을 덮어쓰므로 비교할 실행별로 폴더를 별도 복사해 보관한다. `--diagnostic-audio`의 whole-stream local STT 제한은 유지되므로 remote에서는 `--save-wav`를 쓴다.

확인 항목: 왼쪽/오른쪽/중앙/다채널 재생의 채널 통계; `안녕하세요`와 `익스트림 모드`의 RAW 및 PARTIAL 교정; 같은 말을 실제 두 번 한 경우 보존; 5초 이상 silence 및 음악에서 환각; 단일 발화 secondary 중복; 작은 두 번째 발화 누락; 60초 이상 이어 말하기의 가독성; 첫 자막/갱신 간격; queue backlog/drop; 실제 proxy roundtrip; GUI와 overlay의 같은 utterance 교체/FINAL 유지. `first_channel_silent_other_active` 또는 downmix 상쇄 후보를 확인한 뒤 `--capture-channel mean`을 대조한다.

12. 다음 단계 권장사항 및 재현 방법

최우선은 실제 Windows captured original에 맞춘 사람 검수 reference와 timestamp를 확보하는 것이다. 그 corpus에서 small/base/turbo의 CER와 잘못된 누락을 비교한 뒤 모델 선택을 확정한다. 이어서 window별 단일 음성 손상을 기준으로 separation routing/원본 보완을 실험하고, word timestamp 또는 음성 구간 시간축을 이용해 더 큰 재인식 변화와 문장별 finalization을 검증한다.

모델 5개 재실행 및 GPU 없는 조립 replay:

```bash
python phase3/benchmark_accuracy.py --models base small medium large-v3-turbo large-v3 \
  --before-assembler phase3/output/accuracy_audit/subtitle_assembler_before.py
python phase3/benchmark_accuracy.py --replay-only
python phase4/benchmark_caption_latency.py --server-url http://127.0.0.1:8787
```

`--references verified.json` 형식은 `{"single_A":{"kind":"human_verified","text":"..."},"single_B":{"kind":"human_verified","text":"..."}}`이며 text는 실제 평가 구간(기본 처음 15초)의 정답이어야 한다. duration/input이 cached audio와 다르면 다른 `--output` 디렉터리를 지정해야 한다. 이번 세션의 원본 assembler 복사본은 output 아래 보존했고 baseline commit과 SHA256은 요약 JSON에 기록했다.

[측정 요약](phase3/accuracy_audit_summary.json), [전체 모델 결과](phase3/output/accuracy_audit/benchmark.json), [warm-up 적용 전송 결과](phase3/output/accuracy_audit/latency_small_warm.json), [텍스트 회귀 결과](phase3/output/accuracy_audit/text_regressions.json), [테스트 로그](phase3/output/accuracy_audit/tests.log)에 근거를 보존했다. 생성 WAV/상세 로그/모델은 기존 gitignore 아래 두었다. 시작 git status는 clean이었으며 commit/push는 수행하지 않았다.
