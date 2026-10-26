# R2-B7 Windows GUI live A/B

GUI 화면과 운영 기본값은 그대로이며, 실행 옵션으로만 window/stride를 변경한다.
옵션 생략 시 3초/2초이며 정수 `window > stride > 0`을 요구한다.
`--evidence-dir`과 `--latency-diagnostics`는 지정한 경우에만 기존 파이프라인에 전달한다.

## Windows PowerShell

저장소 루트에서, 이미 준비된 RunPod 서버의 URL을 사용한다. 서버의 모델/디코딩 설정을
R2-B6과 동일하게 유지하고 GUI에서 같은 화자 모드(single), 출력 장치를 선택한다.
아래 명령은 서버를 시작하지 않는다. GUI가 열린 뒤 **시작**을 눌러야 캡처가 시작된다.
각 실행마다 새 evidence 폴더를 사용한다. 같은 프로세스에서 다시 시작하면 같은 경로에
저장되므로 A/B 측정마다 GUI를 종료하고 새 명령으로 실행한다.

```powershell
$env:GPU_SERVER_URL = "https://YOUR-RUNPOD-8787.proxy.runpod.net"
$runA = "phase5\output\r2_b7_A_" + (Get-Date -Format "yyyyMMdd_HHmmss")
.\.venv\Scripts\python.exe -m phase5.gui_app --window-seconds 3 --stride-seconds 2 --evidence-dir $runA --latency-diagnostics
# GUI에서 중지 후 worker 종료를 기다리고 창을 닫은 다음 보관한다.
Copy-Item phase5\output\gui_last_run.json "$runA\run.json"
Copy-Item phase5\output\voxlux.log "$runA\voxlux.log"

$runB = "phase5\output\r2_b7_B_" + (Get-Date -Format "yyyyMMdd_HHmmss")
.\.venv\Scripts\python.exe -m phase5.gui_app --window-seconds 4 --stride-seconds 3 --evidence-dir $runB --latency-diagnostics
# 동일하게 중지/종료한 다음 보관한다.
Copy-Item phase5\output\gui_last_run.json "$runB\run.json"
Copy-Item phase5\output\voxlux.log "$runB\voxlux.log"
```

진단 없는 기존 실행: `.\.venv\Scripts\python.exe -m phase5.gui_app`.
`gui_last_run.json`은 기존처럼 덮어쓰고 `voxlux.log`는 누적 기록한다. 위 명령의 로그 복사본에도
과거 세션이 포함될 수 있으므로 시간/세션 경계를 구분한다. 실행 오류가 있으면 이전 run.json을
현재 결과로 오인하지 않도록 생성 시각과 window/stride를 확인한다.

## 라이브 확인 항목

- 동일 음원을 같은 위치부터 같은 출력 장치/음량으로 재생한다. 시작·종료 여유 무음을 두고,
  저장된 `original.wav`의 실제 포함 구간을 대조한다. 캡처 시작 시점 차이를 STT 누락으로 세지 않는다.
- `run.json`의 window/stride, errors/dropped, RAW/assembly/state/publication 이벤트를 비교한다.
  `timeline.jsonl`, 원본 및 window WAV는 원문 누락·반복을 청취 검수하는 근거다.
- PARTIAL 갱신, FINAL 전환, FINAL 뒤 중복 표시, 중지·재시작, 오버레이 만료와 긴 줄 표시를 확인한다.
  기존 GUI는 중지 요청 이후 이벤트 공개를 억제하므로 종료 직전 FINAL을 자연 발생 FINAL과 구분한다.
- 지연 진단은 캡처 완료→자막 생성 및 HTTP/서버 시간을 기록한다. 실제 Qt 화면에 그려진 시각이나
  단어 발화→화면 표시 지연의 직접 측정은 아니므로 화면/오디오 녹화로 보완한다.
- evidence 저장 I/O는 측정에 영향을 줄 수 있으므로 양쪽에 같은 진단 옵션을 사용한다.

Linux offscreen 단위 테스트는 실제 Windows WASAPI, 데스크톱 합성, RunPod proxy/GPU 검증을 대체하지 않는다.

## 이번 Linux 검증

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s phase5 -p 'test_*.py' -v
PYTHONPATH=.:experiments/r2_b6:phase3:phase4 .venv/bin/python experiments/r2_b6/run_checks.py
git diff --check
```

- GUI: 장치 열거를 mock하도록 테스트 환경을 보강한 후 70개 중 65 PASS, 5 ERROR.
  5개 모두 기존 `phase3/output/evidence/news_20261022_160827/timeline.jsonl` 부재로 실패한다.
  기존 검증 조건은 유지했으며 fixture를 합성하거나 원본 evidence를 수정하지 않았다.
  전체 GUI 회귀 통과는 해당 원본 파일 확보 후 재실행해야 한다.
- Phase 3: 175 PASS, 기존 source-support expected failure 3개.
- Phase 4: 67 PASS. 기존 assertion 스크립트 14개 PASS.
- 실행기에 포함된 R2-B6 단위 검사: 2 PASS, 원본 evidence 부재 1 SKIP. GPU 비교 실험은 실행하지 않았다.
- 수정 Python 파일 AST syntax/import, GUI CLI 도움말, diff 공백 검사 PASS.
- Ruff는 미설치로 SKIP. 실제 Windows WASAPI/GUI/EXE 및 GPU 라이브 검증은 미실행.
