[← README로 돌아가기](../README.md)

# 설치와 실행

## 구성과 검증 범위

Windows 클라이언트는 시스템 오디오 캡처·자막 조립·GUI를, RunPod Linux GPU 서버는 SINGLE VAD/STT를 담당합니다. 아래 명령은 현재 parser와 실행 코드를 확인한 절차입니다. 이 문서 작업에서 설치된 Linux CPU 환경의 테스트는 실행했지만 새 Windows/RunPod 전체 설치나 GPU 모델 다운로드는 수행하지 않았습니다.

Python 3.10 또는 3.11을 권장합니다. 프로젝트의 `requirements.txt`는 직접 의존성을 pin하지만 전체 transitive lockfile은 아닙니다. [runpod-pip-freeze.txt](../runpod-pip-freeze.txt)는 과거 환경 기록이며 모든 플랫폼에 그대로 설치할 lockfile로 취급하지 않습니다.

## Windows 클라이언트

### Python과 가상환경

프로젝트 루트 PowerShell에서 실행합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

이 설치는 저장소의 통합 의존성을 포함하므로 클라이언트만 필요한 경우에도 모델 관련 패키지가 설치될 수 있습니다. remote GUI에서 로컬 CUDA 추론은 수행하지 않지만 Python 의존성 설치 비용과 실행 시 모델 로딩은 다릅니다. 최소 클라이언트 별도 requirements 파일은 현재 없습니다. 로컬 GPU 모드로 전환하지 않는 한 클라이언트 CUDA는 필요하지 않습니다.

직접 의존성은 NumPy 1.26.4, SciPy 1.15.3, SoundFile 0.12.1, SoundCard 0.4.6, requests 2.32.5, websockets 16.1.1, PySide6 6.8.3 등이며 전체 목록은 [requirements.txt](../requirements.txt)를 따릅니다.

### 서버 URL과 GUI 실행

```powershell
$env:GPU_SERVER_URL = "https://YOUR-SERVER-HOST"
.\.venv\Scripts\python.exe -m phase5.gui_app
```

URL은 자신의 서버로 바꿉니다. 코드에 남아 있는 임시 기본 URL의 유효성을 전제로 하지 않습니다. GUI에서도 실행 전 서버 URL과 출력 장치를 선택할 수 있습니다. GUI preflight는 health, 알려진 SINGLE profile 일치, Windows 및 loopback 장치를 확인합니다. 중지 후 worker 종료 전에는 새 실행을 시작하지 않습니다.

운영 키·비밀번호를 URL이나 저장소 파일에 넣지 않습니다. 애플리케이션 자체의 필수 API key는 없습니다.

`RemoteGPUClient`는 `requests.Session.trust_env=False`로 환경변수 기반 프록시와 CA bundle 설정을 자동 사용하지 않습니다. 회사망이나 클라우드 환경에서 `HTTPS_PROXY` 등을 설정하는 것만으로 접속 경로가 바뀌지는 않습니다. 이 경우 실제 클라이언트가 직접 도달할 수 있는 URL과 TLS 신뢰 구성을 확인해야 합니다. PowerShell의 health 요청 성공도 Python 클라이언트의 동일 경로 접속 성공을 보장하지 않습니다. TLS 검증을 끄는 방식으로 우회하지 않습니다.

## RunPod GPU 서버

### 런타임과 패키지

NVIDIA GPU와 드라이버에 맞는 CUDA PyTorch 이미지 또는 환경을 먼저 준비합니다. `requirements.txt`는 이미지의 torch를 바꾸지 않도록 직접 torch를 지정하지 않지만, 일부 transitive dependency는 torch를 요구하므로 설치 전후 버전과 CUDA 가용성을 확인합니다.

```bash
python --version
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

가상환경의 Python 실행 파일이 서버 실행에도 사용되어야 합니다. `soundfile` shared library 오류가 있는 Ubuntu에서는 관리 권한으로 `libsndfile1` 설치가 필요할 수 있습니다. faster-whisper CUDA 실행은 선택한 CTranslate2·CUDA·cuDNN 조합의 호환성도 확인해야 합니다. 임의 버전 조합의 동작을 보장하지 않습니다.

### SINGLE 서버 시작

```bash
python phase4/gpu_processing_server.py \
  --host 0.0.0.0 --port 8787 \
  --processing-profile single \
  --whisper-model large-v3-turbo
```

기본 모델 로드 위치는 `checkpoints/faster-whisper`입니다. 캐시에 모델이 없으면 최초 실행에서 faster-whisper와 Silero 모델 다운로드용 네트워크·디스크가 필요합니다. SINGLE에서는 ClearVoice separator를 로드하지 않습니다. 코드의 일부 공용 import와 통합 requirements에 과거 의존성이 남아 있는 점과 실제 separator 실행은 구분합니다.

기본 warmup을 유지합니다. `--no-warmup`은 cold-start 측정용이며 정상 readiness 대기 시간을 없애는 설정이 아닙니다. `--whisper-model base`는 과거 비교용이고, 이 문서의 기본 운영 경로는 turbo입니다.

서버 기본 bind는 `127.0.0.1`이므로 Windows에서 접속하려면 위처럼 host를 명시하고 RunPod에서 8787 포트에 접근할 경로를 구성해야 합니다. 애플리케이션 서버에는 인증·TLS가 구현되어 있지 않습니다. 실제 외부 노출은 서비스의 HTTPS proxy와 별도 접근 제어를 구성한 뒤 사용합니다. HTTPS proxy 존재만으로 인증까지 제공된다고 간주하지 않습니다.

### 연결과 readiness

클라이언트가 사용할 실제 URL에 대해 확인합니다.

```powershell
Invoke-RestMethod -Uri "$env:GPU_SERVER_URL/healthz"
```

`status=ready`, `gpu_ready=true`, `models_loaded=true`, `processing_profile=single`, `stt_model=large-v3-turbo`를 확인합니다. `model_status.mossformer2_loaded=false`는 SINGLE의 정상 상태입니다. 포트만 열렸거나 프로세스가 있다는 사실은 readiness 검증을 대신하지 않습니다.

## CLI 및 오디오 없이 가능한 확인

Windows의 GUI 대신 remote CLI를 실행하려면:

```powershell
.\.venv\Scripts\python.exe phase3/run_overlap_pipeline.py `
  --live --duration 60 --processing-mode remote `
  --remote-server-url "$env:GPU_SERVER_URL" --speaker-mode single `
  --window-seconds 3 --stride-seconds 2 --assemble `
  --remote-connect-timeout 5 --remote-read-timeout 30 `
  --json-output phase3/output/remote_single.json
```

브라우저 자막 경로가 필요하면 `--websocket`을 추가하고 [phase4/web/index.html](../phase4/web/index.html)을 사용합니다. GUI의 직접 Qt signal 경로에는 필수가 아닙니다.

Linux CPU에서 오디오·모델 추론 없이 가능한 검사:

```bash
python phase4/test_websocket_mvp.py
QT_QPA_PLATFORM=offscreen PYTHONPATH=phase3:phase4:. \
python -m unittest -v phase4.test_remote_gpu_client phase4.test_gpu_processing_server
python phase4/run_websocket_demo.py --count 8 --interval 2
```

테스트는 모의 모델/localhost HTTP를 사용합니다. synthetic WebSocket demo의 성공은 Windows 캡처나 한국어 인식 정확도의 증거가 아닙니다. 전체 검증 명령과 한계는 [성능 문서](performance.md)에 있습니다.

## 환경변수와 파일

| 항목 | 용도 |
|---|---|
| `GPU_SERVER_URL` | GUI가 기본으로 사용할 실제 서버 URL |
| `QT_QPA_PLATFORM=offscreen` | Linux headless 테스트 전용; Windows GUI에서는 설정하지 않음 |
| `PYTHONPATH=phase3:phase4:.` | Linux 루트 실행 테스트의 sibling import 경로 |
| `--evidence-dir` | 명시적으로 원음·window·timeline 근거 저장; 별도 새 디렉터리 사용 |
| `--latency-diagnostics` | 세부 타이밍 기록 |

evidence는 원음과 발화를 포함할 수 있으므로 공개 자료로 옮기기 전에 내용과 경로를 검토합니다. API 키·개인 경로·실제 임시 endpoint는 문서 예제에 넣지 않습니다.

## 연결·실행 오류 진단

| 증상 | 확인 순서 |
|---|---|
| health 연결 실패 | URL·포트 노출·proxy 경로·서버 로그 확인 |
| profile mismatch | 서버 `--processing-profile single`로 실행됐는지 확인 |
| CUDA/cuDNN 오류 | 같은 venv의 torch와 CTranslate2 런타임 조합 확인 |
| 모델 로드 실패 | cache·다운로드 연결·디스크 확인; readiness 전까지 요청하지 않음 |
| 출력은 들리지만 자막 없음 | 출력 장치 ID·loopback·채널 신호·VAD·RAW·drop 순으로 확인 |
| HTTP 503 `server_busy` | 서버 admission 초과 여부와 동시 client·대기 확인 |
| timeout 후 복구 | 해당 창은 재시도하지 않음; 실패 기록과 다음 창 상태 확인 |
| GUI 기록 재생 test 오류 | 원본 timeline fixture 부재 여부 확인; 가짜 파일로 대체하지 않음 |

## EXE 배포 상태

[voxlux.spec](../voxlux.spec)와 [tools/build_windows.py](../tools/build_windows.py)가 PyInstaller onedir 클라이언트 빌드 구성을 제공합니다. 서버·torch·STT 모델을 client bundle에서 제외합니다. [d055e8a](https://github.com/Gwanhoo/Speech-Caption-AI/commit/d055e8a)는 v1.0.1 Windows client 패키징 변경을 보존합니다.

현재 checkout에는 검증할 EXE 배포 산출물과 최신 실행 로그가 없어 **EXE 실행 방법이나 최신 배포 성공을 보장하지 않습니다**. 소스 실행 절차를 우선 사용합니다. EXE 배포 문서화를 완료하려면 동일 버전 artifact·해시·Windows 실행/연결/중지·재시작 결과가 필요합니다.
