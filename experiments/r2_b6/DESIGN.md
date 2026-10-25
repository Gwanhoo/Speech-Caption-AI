# R2-B6 실험 설계

실행 전 조사 기준: HEAD `1d306bd49401610201db61254c433d25d9be0fb2`.

## 연결 구조와 동등성

- `phase3/run_overlap_pipeline.py`: capture의 sample slicing → bounded audio queue → `phase4/remote_gpu_client.py` HTTP → tracker → single primary selection → fragment routing → source admission → audio-overlap evidence → `SubtitleAssembler` → `SpeakerSubtitleState` → publish/UI.
- `phase4/gpu_processing_server.py`: 기존 persistent executor의 `single` 프로필은 원본 파형을 raw slot 0으로 전달하고 slot 1을 0으로 채운다. Silero VAD(최소 200 ms) 통과 시에만 `transcribe_base` 호출. MossFormer2는 로드하지 않는다.
- `phase3/stt_context.py`: ko, beam_size=5, condition_on_previous_text=False, initial_prompt=None. SpeakerPromptContext는 이 운영 경로에서 사용하지 않는다. 나머지는 같은 서버/라이브러리의 기본 디코딩 옵션이다.
- 실험은 기존 HTTP client와 tracker, primary selection, routing, admission, audio-overlap, assembler/state 구현을 재사용한다. assembler minimum=6, sentence lifecycle=True, allow_shifted_head=True, final support=True, silence=2000 ms. 각 후보/반복마다 로컬 상태를 초기화한다. 모델은 기존 서버 인스턴스를 재사용한다.
- 운영 queue/capture/WebSocket/GUI를 실행하지 않는다. HTTP와 로컬 자막 처리 시간을 실측하고, 완료 window 수집 시각에 이를 적용한 직렬 no-drop 가상 시간축을 계산한다. 따라서 Windows/proxy/UI까지의 실제 지연과 동일하다고 주장하지 않는다. 가상 backlog가 생기면 운영 bounded-queue drop과의 차이가 남는다.

## 데이터와 비교 조건

- 원본 WAV: 16000 Hz, mono, FLOAT, 640000 samples, 40초. 저장된 19개 mixed window가 모두 원본 sample slice와 정확히 일치한다.
- 원본 WAV/timeline/run/windows 전체 SHA-256을 실행 전후 검사한다. 결과 폴더는 원본과 분리하고 기존 결과 경로 덮어쓰기를 거부한다.
- 완전 window만 사용한다. baseline=3/2:19개,0–39초; A=4/2:19개,0–40초; B=4/3:13개,0–40초; C=5/3:12개,0–38초. 입력 뒤쪽 1/0/0/2초는 운영 분할에 따른 미처리 구간이며 STT 누락으로 세지 않는다. 공통 포함 범위는 0–38초이나 마지막 window의 추가 context가 달라 완전한 단어별 정렬 비교는 불가하다.
- 기준 설정도 다시 추론한다. 과거 evidence 최상위에 모델 식별자가 없어 과거 RAW와의 차이를 window 변경 효과로 해석하지 않는다.
- 서버 health에서 모델/프로필/옵션/준비 상태를 검사한다. 각 geometry에서 실제 음성 1회 warm-up 후 seed가 고정된 순서로 설정당 3회. 제출/성공 요청 수와 모델 load count를 검사해 동시 외부 요청·모델 재로드를 탐지한다.
- RAW, admitted text, VAD, mapping, overlap, assembly event, PARTIAL/FINAL, HTTP/server/local timing, 가상 latency, coverage를 window별 보존한다.
- 의심 지표: VAD positive/empty RAW, admission rejection, 미병합 fuzzy 경계, assembler suspicious deletion, 이전 RAW의 6글자 구간이 session에 없는 경우, FINAL 간 경계 반복. 모두 검토 후보이며 정답 오류 수가 아니다.
- R2-B3/B4 이름으로 된 별도 audit는 이 checkout에서 찾지 못했다. 기존 news fixture의 중복/수정 후보와 원본 FINAL 경계를 시간 구간으로 대조한다. 정답 전사본, CER/WER, 음향 단어 정렬은 사용하지 않는다.

## 사전 검증

`PYTHONPATH=experiments/r2_b6:phase3:phase4 .venv/bin/python -m unittest discover -s experiments/r2_b6 -p 'test_*.py' -v`

3 tests PASS: 분할/미처리 tail, 잘못된 모델·옵션 차단, 실제 evidence의 admission 일치 및 기존 '실행하거나' 보존 회귀. 기존 Phase 3/4 회귀 테스트는 측정 후에도 실행한다.

## 실행

```bash
.venv/bin/python experiments/r2_b6/compare_windows.py \
  --output experiments/r2_b6/results --repeats 3
```

원본·운영 설정 변경, 패키지 설치, 모델 다운로드, RunPod 자원 시작, commit/push를 수행하지 않는다.
