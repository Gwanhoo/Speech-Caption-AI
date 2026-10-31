# VOXLUX 정확도·안정성 검증 — 2026-10-31

> **후속 H1/M1 검증으로 아래 실험의 새 어절 경계 정정 분기를 철회했다.**
> 현재 Assembler는 `b53b208`과 동일하며 H1의 두 정상 문장을 보존한다.
> 아래의 A 정정 성공·CPU 비교·207 tests 수치는 철회 전 실험 기록이다.
> 현재 A의 조기 FINAL 및 M1은 미해결이다. 최신 코드 판단·테스트·지연 수치는
> [H1/M1 후속 보고서](h1_m1/REPORT.md)를 따른다. 기존 증거 파일은 보존했다.

## 결론과 증거의 범위

기존 Assembler의 **어절 경계에서 달라진 단일 서술부를 후속 창으로 정정하지 못하고 FINAL로 넘기는 경로**를 좁게 보완했다. STT 설정, 전체 fuzzy 임계값, State/Publication 계약, GUI 및 원격 프로토콜은 변경하지 않았다. 모든 FINAL을 지연시키거나 이미 확정한 문장을 수정하지 않는다.

사용자 제공 사례 A의 RAW 두 창과 합성 overlap 근거에서는 잘못된 FINAL이 1→0이다. 다른 어휘의 합성 변형 2개에서도 1→0이다. **10월 31일 original.wav/timeline/gui_trace가 없으므로, 실제 사례 A의 acoustic 조건 충족이나 실시간 인식 정확도 개선은 아직 입증되지 않았다.** 수정은 `shared_speech=True`와 `supported_tail_revision=True`일 때만 새 경로를 허용한다. 해당 실행에서 이 조건이 False/unknown이면 사례 A는 여전히 미해결이다.

원본이 있는 과거 A/B WAV로 RTX A5000에서 beam 1/3/5를 비교했다. 사람 검수 정답이 없는 별도 자료이며 최신 뉴스 오디오가 아니다. RAW와 FINAL을 각각 저장했지만 CER/WER, 환각·누락·문장 경계 오류 횟수는 null로 남겼다. 모델이 생성한 문장을 정답으로 사용하지 않았다.

## 1. 작업 시작·종료 환경과 Git

요청된 사전 명령의 실제 도구 출력:

- `git status --short`: 출력 없음, 시작 시 clean. 기존 수정/untracked 없음.
- `git branch --show-current`: `main`.
- `git log -1 --oneline`: `b53b208 fix: limit overlay to three visual lines and prioritize recent captions`.
- 전체 HEAD: `b53b208fa2e5beabad097c675ff231838f808cb5`.
- `git ls-remote origin refs/heads/main`: 동일 해시. checkout 시점의 실제 GitHub main 일치를 읽기 전용 확인했다.
- `nvidia-smi` (07:34:15 UTC): RTX A5000, 24,564 MiB, 2,311 MiB 사용, GPU 0%; driver 595.91.07, CUDA driver 13.2. Python 3.12.3.
- 기존 PID 6793: `python -u phase4/gpu_processing_server.py --host 0.0.0.0 --port 8787 --whisper-model large-v3-turbo --processing-profile single`.

AGENTS.md, README, REMOTE_GPU.md 및 P0 정확도/ghost/회귀 자료를 확인했다. 기본 sandbox는 kernel namespace 제한으로 실행 불가능하여 승인된 escalation으로 명령을 실행했다. 시스템 설정은 바꾸지 않았다. 테스트 의존성 websockets/ruff만 `/tmp/voxlux-accuracy-deps`에 설치했다.

종료 HEAD/branch는 동일. 수정 파일 3개와 `docs/accuracy_20261031/`의 신규 검증 자료가 남는다. commit/push, reset/clean, 운영 서버 중지·재시작, 모델 다운로드, 새 RunPod 자원 시작은 하지 않았다. 운영 PID는 유지되고 GPU probe 프로세스 종료 후 VRAM은 다시 2,311 MiB, GPU 0%였다.

## 2. 실제 코드 경로

| 경로 | 확인한 구현 및 함의 |
|---|---|
| STT | `phase4/gpu_processing_server.py:313` `_process_serialized`: single은 원본 slot0과 inactive slot1, separator 생략. VAD 양성 창 전체를 `transcribe_base`에 전달. `phase3/stt_context.py:10`: ko, beam5, previous-text false. lazy segments를 materialize하는 시간까지 STT timing에 포함. confidence는 진단값이며 의미상 정답 보증이 아님. |
| 서버 생명주기 | persistent 모델, warmup, serialized inference/worker 및 busy 처리. 서버 변경 없음. |
| remote | `phase4/remote_gpu_client.py:120` 결과 identity/배열/VAD 계약 검증, timeout·취소·retry 없는 요청 경로. 서버 timing과 client roundtrip/parse는 분리. 서로 다른 프로세스의 절대 monotonic 시각을 빼지 않음. |
| 큐·drop | `run_overlap_pipeline.py:1081` STOP/stale 구분, `:1486` 늦은 remote 응답은 자막 상태 갱신 전에 폐기. 기존 bounded queue와 drop accounting을 유지. |
| overlap 근거 | `secondary_leakage_diagnostics.py:542` `subtitle_overlap_evidence`: 인접 VAD 교집합 → shared_speech; tail coverage geometry + waveform correlation → supported_tail_revision. 이것은 단어별 acoustic alignment가 아님. |
| Assembler | `subtitle_assembler.py`: exact/relaxed/fuzzy/anchored/supported correction 순서, confirmed prefix 보호, FINAL history 보호. 미일치 + 이전 끝 문장부호이면 sentence rollover. 이번 결함은 이 fallback 전에 안전한 어절 경계 정정을 놓치는 부분. |
| State·Consensus | `SpeakerSubtitleState.process`: assembler의 confirmed_prefix_length를 독립 관측 근거로 사용. 단순 누적 prefix를 새 독립 관측으로 취급하지 않음. `finalize`: require_final_support이면 `_source_supported_text`만 확정. |
| Publication | `run_overlap_pipeline.py:2150` 부근: rollover일 때 이전 state FINAL→assembler commit→새 hypothesis process. 수정에서 new_fragment는 현재 RAW 전체이므로 기존 source coverage 계산을 그대로 사용. unsupported copied prefix를 새 source support로 세탁하지 않음. |
| 무음 | `SpeakerSubtitleState`: 마지막 speech 창의 stream end와 현재 창 end 차이로 기본 2초 무음 FINAL. VAD onset/offset 단위의 정확한 발화 종료 시각이 아님. |

## 3. A-3 오류 유형별 분석

아래의 합성 텍스트·합성 파형은 알고리즘 경로의 증거다. 음성 인식 품질이나 사용자의 실제 발음 원인을 증명하지 않는다.

| 유형 | 발생 조건·재현 근거 | 책임/코드 | 기존 방어 | 결정 및 회귀 위험 |
|---|---|---|---|---|
| 1. 환각·음향 미지원 문장 | 사용자 A는 정답과 RAW 불일치. 오인식인지 hallucination인지 음원 없이 구분 불가. 기존 ghost/source tests는 unsupported publication 경로 재현 | STT + Publication; `stt_context.transcribe_base`, `SpeakerSubtitleState.finalize` | VAD inactive STT skip, admission/source coverage, confidence diagnostics | decoder threshold 변경 보류. 강화 시 조용한 실제 말 삭제 위험. 정정 가능한 잘못된 PARTIAL만 이번 범위 |
| 2. 짧은 창 문맥 손실 | 3/2초 창과 previous-text false는 코드로 확인. 손실의 인과관계는 동일 원음/정답 A/B 필요 | STT; `stt_context`, 기존 `benchmark_stt_context.py` | persistent 모델, configurable window/stride/context 실험 도구 | 설정 유지. 창 확대는 첫 자막/수집 지연, prompt는 오염 위험 |
| 3. 경계 음절·어절 누락 | 사용자 B: 공교통/대중교통 경계. 입력 RAW 재생에서 정상 prefix 유지 확인. D의 정확한 RAW는 미제공 | STT + Assembler; suffix-prefix, boundary exact, terminal stem | clipped head, terminal stem completion, unobserved prefix 보호 | 기존 동작 보존. B 신규 회귀 PASS, D 합성 부정·시제 및 기존 recorded tests PASS. 무조건 최신 RAW 선택 시 정상 음절 소실 |
| 4. overlap 중복·누적 손상 | 기존 finalized-tail 재등장 회귀 및 production fixtures | Assembler; `commit_finalized_text`, `_finalized_overlap_tail` | adjacent/shared exact tail, reset/gap invalidation, active/session 분리 | 변경 안 함. 동일 문장 실제 반복을 전역 중복 필터로 제거하지 않음 |
| 5. 잘못된 fuzzy 교체 | 사용자 C의 지역의→지역에와 기존 unsupported replacement/confirmed prefix 사례 | Assembler; `corrected_suffix_prefix`, `is_internal_deletion_revision` | fuzzy 0.88, deletion-only 거부, supported terminal revision, confirmed prefix 보호 | 전체 fuzzy 변경 없음. C PASS. 새 분기는 `match_type=none`에서만 실행; 광범위 fuzzy 강화는 정상 어휘 삭제 위험 |
| 6. PARTIAL 조기 FINAL | A + 도서관/연구모임 합성 변형: 기존 final+partial, 수정 partial. `before_revision.txt` 실패 3개 | Assembler rollover → State finalize | 어미 내부 정정, supported short tail anchor | 최소 수정 적용. 강한 overlap + 여러 어절 anchor + 단일 짧은 미확정 서술부. 반복 주제가 실제 새 문장일 가능성은 남으므로 live 확인 필요 |
| 7. 문장부호 오분리 | 마침표 후보 + match none이 `sentence_boundary_confirmed`로 확정. A anchor는 완성 어절이고 기존 tail 길이/어절 내부 조건에 불합격 | Assembler; `has_sentence_ending`, `reconcile_adjacent_sentence_head_revision`, rollover | supported ending revision/기존 4자 이내 어미 정정 | 문장부호 전면 무시 대신 근거 있는 revision을 우선. 근거 없는 unrelated 새 문장의 FINAL 시점은 유지 |
| 8. VAD silence/FINAL 불일치 | 2초 간격 창에서 whole-window speech flag와 end 기준 타이밍 사용. 최신 실제 VAD timeline 없음 | State; `process` silence branch | last_speech_time, 2초 threshold, reset | 재현된 새로운 결함 없음, 보류. 무음→음성→무음 tests PASS. threshold 축소는 발화 중간 확정 위험 |
| 9. 무음 유령 자막 | 기존 VAD-positive artifact, transient 316/540ms, residual/source regression으로 재현 가능 | VAD/Admission/Publication; `secondary_leakage_diagnostics`, `test_ghost_subtitle_regressions`, Phase4 ghost tests | silence gate, source support, candidate hold/discard, raw/VAD 계약 | 기존 방어 유지. 기존 noise/projection XFAIL 2건 미해결. 새로운 임계값 추측 변경은 정상 짧은 말 억제 위험 |
| 10. 정상 음성 누락·과억제 | source-supported prefix gap 뒤 새 suffix가 비공개인 기존 XFAIL. 200ms VAD minimum도 확인되지만 현재 실제 누락률 없음 | Admission/Publication; `test_source_support_diagnostics`, VAD helper | 약한 실제 음성 보존, provenance와 source gap 보호 | 기존 XFAIL 1건 미해결. guard 제거 시 unsupported prefix가 공개되므로 이번 변경 제외. 짧은 `네.`·반복·FINAL 이후 발화 PASS |

## 4. 최소 수정 및 회귀 보호

운영 변경은 `phase3/subtitle_assembler.py` 하나다. 기존 `reconcile_adjacent_sentence_head_revision`에 optional keyword `supported_tail_revision=False`를 추가하고 caller의 기존 evidence를 전달했다.

새 어절 경계 분기의 모든 조건:

1. sentence lifecycle 활성, 기존 완료 후보, 아직 active, 바로 인접 창, `shared_speech is True`, 기존 exact/fuzzy/교정이 처리하지 못한 입력.
2. 기존 VAD tail coverage와 파형 상관 근거인 `supported_tail_revision=True`.
3. current head가 previous 내부에 고유하게 한 번만 나타남. 양쪽 모두 완전한 어절 경계이며 최소 기본 6 정규화 문자와 2개 어절 일치.
4. old suffix는 문장부호로 끝나는 **한 어절**이고 길이가 anchor 이하. current의 새 내용은 anchor 이상. anchor 내부 또는 이후의 독립 문장·여러 어절 old assertion·짧은 변경·반복 anchor 거부.
5. active suffix 일치, seam이 confirmed prefix 밖. prefix와 FINAL history는 그대로 보존.

새 분기도 기존 `sentence_head_revision` 이벤트를 사용한다. 정정된 PARTIAL과 기존 source-supported publication을 같은 창에서 처리한다. 창 추가 대기, GUI 이벤트 필드 추가, remote 필드 변경, model option 변경은 없다.

테스트 변경:

- `phase3/test_general_subtitle_lifecycle.py`: A/일반화, evidence 없음·gap, B/C 및 D 합성, normal boundary, repeated topic/anchor, short reply, silence cycle, unsupported publication, finalized history 불변성.
- `phase3/test_live_caption_continuity.py`: 실제 기존 overlap 함수에 합성 waveform/VAD를 넣어 coverage/correlation 실패 시 정정하지 않는지 검사.
- 과거 production fixture, 연결 어미, `않습니다/않았습니다` 계열·supported/unsupported replacement·remote STOP tests는 기존 검증을 그대로 실행했다.

## 5. A-4 GPU 비교

`gpu_probe.py`는 운영 서버의 8787 established connection이 없고 GPU 여유가 충분한지 확인한 뒤 시작하며, 창마다 연결 발생을 확인한다. 캐시된 large-v3-turbo float16 인스턴스 한 개만 추가 로드했다. 다른 모델 캐시는 없어 추가 후보 다운로드는 SKIP했다. 원격 endpoint 추론 호출이나 서버 설정 변경은 없다.

- 입력: 기존 `phase3/test_audio/A.wav`, `B.wav`, 각 첫 6개 완전한 3초/2초 창(각 0–13초). 원본 SHA256은 `gpu_probe.json`.
- 원본 24kHz mono를 scipy rational 2/3로 16kHz 변환. Silero VAD 200ms gate, ko/previous-text false 유지. beam마다 warmup 후 두 번 반복, 순서 5/1/3 및 3/1/5.
- 총 72회 실 GPU decode, beam당 24회. VAD/Assembler/State 재생 결과를 RAW와 분리 저장. 실제 WASAPI/remote/GUI end-to-end 경로는 아님.

| beam | STT 평균 | p50 | p95 | 관측 process VRAM 최대 |
|---|---:|---:|---:|---:|
| 5 기준 | 0.1926s | 0.1900s | 0.2175s | 2,302 MiB |
| 1 | 0.1521s | 0.1539s | 0.1740s | 2,302 MiB |
| 3 | 0.1695s | 0.1669s | 0.1882s | 2,302 MiB |

VRAM은 창 처리 후 nvidia-smi snapshot 최대이며 순간 peak 보장이 아니다. 기존 서버 VRAM과 별개 프로세스 수치다. 모델 unload 직후 CUDA context 약 218 MiB가 남았지만 probe 프로세스 종료 후 총 사용량은 시작과 같은 2,311 MiB였다. 실제 뉴스 테스트 평균 STT 0.137초와 위 값은 입력/조건이 달라 직접 비교할 수 없다.

beam에 따라 RAW/FINAL 표현이 달랐다. 정답 없이 어느 쪽이 더 정확한지 선택하지 않으며 **운영 beam 5 유지**. 누락 음절·어절, 환각, 문장 경계 오류, RAW/FINAL CER/WER는 미측정(null). 같은 GPU RAW와 VAD를 baseline/수정본에 각각 재생한 12 runs에서 FINAL은 12/12 동일(`gpu_replay.json`); 실제 음성에서 개선을 입증한 것이 아니다.

### 채점 정규화

텍스트 회귀는 기존 `phase4.benchmark_stt_models.score_transcript`를 재사용했다. CER: 완성형 한글 `[가-힣]`, ASCII 영문/숫자만 유지, 영문 lowercase, 공백·문장부호 삭제 후 Levenshtein/reference 문자수. WER: 같은 문자 집합의 연속 run을 어절 토큰으로 취급(`re.findall`), 영문 lowercase, token Levenshtein/reference 토큰수. 숫자 읽기·띄어쓰기 의미 통합·발음 치환은 하지 않는다. 모든 텍스트 비교에 동일 규칙을 적용했다. 이후 실제 음성 평가에도 동일 규칙을 사용하되 원본과 일치하는 사람 검수 정답을 먼저 확보해야 한다.

## 6. 변경 전후 비교

`compare.py`는 `git show`로 baseline 코드를 메모리에 로드한다. checkout을 변경하지 않는다. 표의 정답은 테스트용 기대 텍스트이며, **음성 인식 CER/WER가 아니다**. 사례 A의 채점 구간은 제공된 W003/W004까지다. 전체 원문 끝의 `나섰습니다`는 별도 follow-up 회귀에서 검증한다.

| 텍스트 재생 | 잘못된 초기 FINAL 전→후 | 조립 FINAL CER 전→후 | WER 전→후 |
|---|---:|---:|---:|
| A, overlap 근거를 합성 주입 | 1→0 | 56.52%→0% | 42.86%→0% |
| 도서관 합성 변형 | 1→0 | 52.00%→0% | 37.50%→0% |
| 연구모임 합성 변형 | 1→0 | 36.36%→0% | 30.00%→0% |
| B / C / D 합성 연결 대조 | 0→0 | 각각 0%→0% | 각각 0%→0% |

정상 대조 3/3의 기대 FINAL 전체 보존, 새 중복 삽입 없음. 이 비율은 작은 텍스트 fixture의 보존 결과이며 실제 정상 발화 보존율·중복률로 일반화할 수 없다. RAW 입력은 전후 동일하고 STT 코드는 바꾸지 않았다.

두 speech 창+silence finalization 한 시나리오의 CPU 비용: warmup 20회 제외, 전후 각각 1,800회 interleaved. 평균 **0.7567→0.7724ms**, p50 **0.8125→0.8480ms**, p95 **1.1033→1.1536ms**. 차이는 평균 +0.0157ms, p95 +0.0503ms다. 다른 테스트 프로세스가 종료된 후 측정했지만 작은 조건 검사 비용 외에도 스케줄링 잡음이 포함되므로 성능 개선/퇴보의 통계적 주장으로 해석하지 않는다. 수집 창/요청을 추가하지 않아 이 재생에서 실시간성을 크게 악화시키는 근거는 없다. Windows 전체 latency의 변화는 별도 실측이 필요하다.

| 실시간 지표 | 사용자 제공 10/31 baseline | 변경 후 |
|---|---:|---|
| 평균 STT | 0.137s | 동일 세션 재실행 없음 |
| 평균 post-capture | 1.493s | 미측정 |
| 첫 자막 | 5.940s | 미측정 |
| steady interval p95 | 4.072s | 미측정; post-capture p95와 다른 지표 |
| post-capture p95 | 미제공 | 미측정 |
| audio queue 최대 | 1 | 실제 remote 세션 미측정 |
| 일반 처리 오류 | 0 | 실제 remote 세션 미측정 |
| STOP drop | 2 | 실제 remote 세션 미측정; 자동 STOP 회귀 PASS |
| 실음성 RAW/FINAL CER/WER·보존율·중복률 | 검수 정답/로그 없음 | 미측정 |

## 7. 실행 테스트

| 실행 | 결과 | 파일 |
|---|---|---|
| 변경 전 Phase3 | 196 실행: 193 PASS, 기존 XFAIL 3 | baseline_phase3.txt |
| 새 실패 재현(수정 전) | 8 methods, 3 assertion FAIL(일반화 subtest 포함) | before_revision.txt |
| 관련 lifecycle/production/시제 | 72 PASS | targeted.txt |
| VAD/waveform gate 포함 continuity | 13 PASS | acoustic_gate.txt |
| 최종 Phase3 전체 | **207 실행: 204 PASS, 기존 XFAIL 3** | phase3_final.txt |
| Phase4 Client/Server 전체 | **88 PASS**, timeout/cancellation/STOP/stale/queue accounting 포함 | phase4_final.txt |
| main-style 기존 회귀 | **9 scripts PASS** | scripts_final.txt |
| 실제 GPU 제한 beam probe | **72 decode 완료**, 프로세스 종료 정상 | gpu_probe.txt/json |
| 정적 검사 | ruff E9/F63/F7/F82, 실험 스크립트 F 검사, Python 3.10 문법 parse, Python 3.12 compile/import, git diff --check PASS | static_final.txt |

headless runner의 soundcard mock은 장치 import에만 적용한다. 실제 WASAPI 성공으로 계산하지 않는다. 기존 테스트 삭제·기대값 약화 없음. 초기 신규 실패는 보존했고 수정 후 재실행했다. 최종 unexpected FAIL/ERROR는 0이다.

SKIP: Windows WASAPI/실제 PySide6 화면/EXE 및 네트워크 end-to-end 재측정(Linux 환경, 최신 Windows evidence 부재). GUI/패키징 코드는 변경하지 않았다. Python 3.10 문법 검증은 실제 Windows Python 3.10 실행 검증을 대체하지 않는다. 다른 모델 비교는 캐시 부재로 다운로드하지 않았다.

## 8. 미해결 항목

- 실제 사례 A의 VAD tail coverage/correlation과 confirmed prefix 미확인. 새 조건이 충족되는지 최신 timeline으로 확인해야 한다. 현재 관찰만으로 음향 hallucination 원인을 확정할 수 없다.
- 강한 VAD overlap도 단어 alignment가 아니다. 같은 여러 어절의 주제를 곧바로 다시 시작하는 발화와 오인식 revision이 모호할 수 있다. 짧은 anchor/반복 anchor/독립 문장/무음/gap guard는 있지만 모든 자연 반복을 구별한다는 보장은 없다.
- 긴 서술부·여러 어절·근거 부족의 오인식은 의도적으로 정정하지 않는다. confirmed 부분의 잘못된 STT 및 FINAL 자체의 사후 교정도 이번 범위 밖이다.
- 기존 XFAIL 3: `unresolved_noise_above_existing_bound`, `unresolved_projection_artifact`, `unresolved_gap_must_not_hide_later_supported_speech`. 각각 허위 source support 2경로와 unsupported gap 뒤 정상 suffix suppression이다. 해결/PASS로 보고하지 않는다.
- RAW 오인식, 200ms 미만 발화, 빠른 경계 음절, 실제 반복/중복, STOP·timeout로 버린 음성의 누락률은 원음과 검수 정답이 필요하다.
- 실제 D W012/W013 RAW는 미제공이므로 정확한 해당 세션 재생은 못 했다. 과거 recorded·synthetic 부정/시제 회귀는 통과했다.

## 9. Windows 실시간 재검증

수정된 client 코드를 Windows 작업 사본에 반영하고 기존 GPU 서버와 single/large-v3-turbo/beam5/3초·2초 설정을 유지한다. 이번 운영 변경은 client Assembler에 있다.

```powershell
$run = "phase3/output/evidence/accuracy_" + (Get-Date -Format "yyyyMMdd_HHmmss")
python -m phase5.gui_app --window-seconds 3 --stride-seconds 2 `
  --evidence-dir $run --latency-diagnostics
```

1. GUI single 및 서버 준비 상태 확인 후, 동일 원본 WAV를 같은 출력 장치/음량으로 재생한다. 최신 뉴스뿐 아니라 대화·짧은 응답·빠른 조사/연결어미·부정/시제·같은 문장 반복·비슷한 주제의 새 문장을 포함한다. baseline b53b208과 수정본을 **각각 별도 작업 사본·evidence 폴더**에서 같은 WAV로 반복 실행한다. 기존 작업 트리에 reset/clean하지 않는다.
2. 5초 무음→발화→5초 무음, 자연 FINAL 이후 재발화, request 처리 중 STOP을 별도 실행한다. STOP drop과 비정상 timeout/stale drop을 구분한다. 정상 실행은 마지막 무음 후 Stop, 강제 종료하지 않는다.
3. `$run/original.wav`, windows WAV, timeline.jsonl, gui_trace 파일, gui_last_run 요약을 보존한다. 원본 음성과 일치하는 사람 검수 정답 및 window별 시간 정답을 함께 만든다. 모델 출력이나 뉴스 예상 대본을 자동 정답으로 쓰지 않는다.
4. 사례 A의 해당 창에서 `audio_overlap_evidence`, `confirmed_prefix_length`, `match_type=sentence_head_revision`, rollover_text, utterance_id, source support를 확인한다. 근거 없는 문장 병합이 없고 기존 `날씨였습니다`가 FINAL로 남지 않는지 확인한다. 조건 불충족 시 그 증거를 보존해 후속 분석한다.
5. B/C의 정상 정정, D의 부정·시제, 같은 문장을 의도적으로 다시 발화했을 때의 두 FINAL, short reply의 생존을 사람이 확인한다. 불일치가 있으면 transcript/시간 구간과 함께 기록한다.
6. RAW CER/WER는 window별 정답으로 채점한다. overlap 창 RAW를 단순 연결해 전체 정답과 비교하지 않는다. 최종 CER/WER는 시간순 FINAL을 utterance별 한 번만 연결해 채점한다. PARTIAL들을 모두 이어 붙이지 않는다. 환각/누락·경계 오류는 청취와 시간 정답으로 별도 집계한다.
7. 정상 발화 보존율, 중복/잘못된 FINAL 횟수, 평균/p95 post-capture, 첫 자막, steady interval, backlog/drop/failure, VRAM을 같은 표본·동일 조건에서 비교한다. CPU/remote/GUI 단계는 각 프로세스 duration과 event identity로 연결한다.

## 10. 변경 파일·재실행·커밋

운영: `phase3/subtitle_assembler.py`.
테스트: `phase3/test_general_subtitle_lifecycle.py`, `phase3/test_live_caption_continuity.py`.
신규 문서/증거/실험 스크립트: `docs/accuracy_20261031/`.

```bash
PYTHONPATH=/tmp/voxlux-accuracy-deps python docs/p0_2_evidence/run_regressions.py phase3
PYTHONPATH=/tmp/voxlux-accuracy-deps python docs/p0_2_evidence/run_regressions.py phase4
python docs/p0_2_evidence/run_script_regressions.py
python docs/accuracy_20261031/compare.py
# 별도 GPU probe는 운영 연결 없음/유휴 GPU/로컬 모델 조건에서만:
HF_HUB_OFFLINE=1 python docs/accuracy_20261031/gpu_probe.py
```

커밋하지 않았다. 신규 커밋 해시 없음. 검증한 기반 HEAD는 위 b53b208 전체 해시다. 이번 최소 코드 수정·자동 회귀는 완료했지만, 기존 XFAIL과 실제 Windows 음성 품질 검증은 미완료로 남긴다.
