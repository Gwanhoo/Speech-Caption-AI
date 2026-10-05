# Speech Caption AI

2026-09-30 정확도 audit, 모델 5개 비교, 자막 교정 및 Windows 검증 항목은
[ACCURACY_AUDIT.md](ACCURACY_AUDIT.md)에 정리했습니다. 새 remote 서버 기본값은
`large-v3-turbo`이며 `--whisper-model base`로 이전 STT 기준선을 재현할 수 있습니다.

Windows capture와 RunPod GPU inference를 분리한 remote mode 실행 방법 및 HTTP 계약은
[phase4/REMOTE_GPU.md](phase4/REMOTE_GPU.md)에 정리되어 있습니다. 기본 실행 모드는 기존 local GPU입니다.

Phase 0-A는 두 개의 한국어 WAV 파일을 같은 시작 시점부터 겹쳐 `mixed.wav`를 만들고, ClearVoice의 `MossFormer2_SS_16K` 모델로 2개 음성 파일을 분리하는 최소 검증 단계입니다. Phase 0-B는 분리된 WAV 파일을 로컬 Whisper 모델로 한국어 텍스트로 변환합니다. Phase 0-C는 두 단계를 한 명령으로 실행합니다.

Phase 4-A에는 Python WebSocket broadcaster와 로컬 브라우저 자막 UI MVP가 추가되어 있습니다. FastAPI, Next.js, VAD, Speaker Tracking은 아직 구현하지 않습니다.

## Phase 5: Windows 데스크톱 GUI

Phase 5 GUI는 기존 Phase 4-I remote live pipeline을 그대로 호출하는 PySide6
frontend입니다. Windows GUI thread와 분리된 worker에서 RunPod health 확인, WASAPI
loopback 장치 확인, capture 및 pipeline 실행을 수행합니다. 자막은 console 문자열을
파싱하지 않고 `SubtitleStateEvent`에서 만들어진 structured event를 Qt signal로
전달합니다. 내부 speaker별 상태는 유지하면서 화면에는 화자 라벨 없이 하나의
chronological subtitle feed로 표시합니다.

Windows PowerShell에서 의존성을 설치하고 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m phase5.gui_app
```

기본 RunPod URL은 `GPU_SERVER_URL` 환경변수로 설정하며, GUI의 서버 URL 입력란에서
실행 전에 한 번만 변경할 수도 있습니다. `중지`는
강제 thread 종료를 사용하지 않고 capture stop event를 설정한 뒤 기존 queue와 worker가
drain/join되기를 기다립니다. GUI 실행 중에도 기존 WebSocket endpoint
`ws://127.0.0.1:8765`가 함께 제공됩니다.

```powershell
$env:GPU_SERVER_URL = "https://YOUR-RUNPOD-8787.proxy.runpod.net"
.\.venv\Scripts\python.exe -m phase5.gui_app
```

오디오/네트워크를 사용하지 않는 headless GUI 테스트:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m unittest -v phase5.test_gui_app
```

## Phase 4-A: WebSocket Subtitle UI

Phase 3-E overlap pipeline을 WebSocket UI와 함께 실행합니다.

```powershell
.\.venv\Scripts\python.exe phase3\run_overlap_pipeline.py `
  --duration 30 `
  --window-seconds 3 `
  --stride-seconds 2 `
  --assemble `
  --websocket `
  --json-output phase3\output\phase4_a_e2e.json
```

WebSocket endpoint is `ws://127.0.0.1:8765`. Start the pipeline first, then open `phase4/web/index.html` in a local browser. The UI keeps the four newest incremental subtitle events for each speaker and reconnects automatically if the engine restarts.

For a WebSocket-only UI check without GPU inference:

```powershell
.\.venv\Scripts\python.exe phase4\run_websocket_demo.py
```

Then open `phase4/web/index.html` while the demo is running.

## Python 버전

- Python 3.10 또는 3.11 권장
- GPU 사용 시 PyTorch/CUDA 조합은 사용 중인 NVIDIA 드라이버와 CUDA 버전에 맞춰 설치해야 합니다.

## 가상환경 생성

Windows PowerShell 기준:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## Dependency 설치

CPU 환경에서 먼저 설치를 확인할 때:

```powershell
pip install numpy scipy soundfile clearvoice
```

NVIDIA GPU를 사용할 경우 `clearvoice` 설치 전에 PyTorch 공식 안내에 맞는 CUDA 버전의 `torch`를 먼저 설치하는 것을 권장합니다.

예시는 환경마다 달라질 수 있으므로, 본인 CUDA/드라이버에 맞는 명령을 PyTorch 공식 설치 페이지에서 확인해야 합니다.

```powershell
pip install numpy scipy soundfile
# 그 다음 CUDA 환경에 맞는 torch 설치
pip install clearvoice
```

## 입력 파일 위치

다음 두 파일을 직접 넣어주세요.

```text
phase0/input/speaker_a.wav
phase0/input/speaker_b.wav
```

입력 WAV는 sample rate나 stereo/mono가 달라도 `mix.py`에서 mono/16kHz로 변환합니다.

## 실행

```powershell
.\.venv\Scripts\python.exe phase0\run.py
```

실행 순서:

```text
입력 파일 검사
-> mixing
-> phase0/output/mixed.wav 생성
-> MossFormer2_SS_16K speech separation
-> phase0/output/speaker_1.wav 저장
-> phase0/output/speaker_2.wav 저장
-> faster-whisper STT
-> phase0/output/speaker_1.txt, speaker_2.txt 저장
```

입력 파일이 없으면 다음처럼 안내하고 종료합니다.

```text
phase0/input/speaker_a.wav와 speaker_b.wav를 넣어주세요.
```

## 출력 파일

```text
phase0/output/mixed.wav
phase0/output/speaker_1.wav
phase0/output/speaker_2.wav
phase0/output/speaker_1.txt
phase0/output/speaker_2.txt
```

## Phase 0-B: 분리 음성 STT

기존 `.venv`에 `faster-whisper`를 설치한 후, 분리된 WAV 두 파일을 각각 변환합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install faster-whisper
.\.venv\Scripts\python.exe phase0\transcribe.py
```

기본값은 다국어 Whisper `base` 모델의 CPU `int8` 실행입니다. 첫 실행 시 모델을 `checkpoints/faster-whisper`에 다운로드합니다. CUDA 12와 cuDNN 9 런타임이 준비된 환경에서는 `--device cuda`로 GPU `float16` 실행을 선택할 수 있습니다. 결과는 콘솔과 UTF-8 텍스트 파일 `phase0/output/speaker_1.txt`, `phase0/output/speaker_2.txt`에 저장됩니다.

## CPU/GPU 관련 사항

- `MossFormer2_SS_16K`는 딥러닝 기반 speech separation 모델이므로 CPU에서는 매우 느릴 수 있습니다.
- Phase 0-A의 목적은 품질 검증이지만, 이후 실시간 시연까지 고려하면 NVIDIA GPU 환경 검증이 중요합니다.
- 최초 실행 시 ClearVoice가 pretrained model checkpoint를 다운로드할 수 있으므로 인터넷 연결이 필요할 수 있습니다.
- Windows에서는 PyTorch CUDA, FFmpeg, libsndfile 관련 환경 문제가 발생할 수 있습니다. WAV만 처리하는 이번 단계에서는 FFmpeg 의존도를 최소화했습니다.

## 프로젝트 구조

```text
phase0/                 WAV 혼합, 음성 분리, 기본 STT 검증
phase1/                 Windows 오디오 캡처와 분리/STT 연결
phase2/                 CUDA 및 연속/청크 처리 벤치마크
phase3/                 실시간 분리/STT, 화자 추적, 자막 조립
phase4/                 WebSocket broadcaster와 브라우저 자막 UI
phase5/                 PySide6 Windows 데스크톱 GUI와 pipeline controller
checkpoints/             실행 시 다운로드되는 모델(버전 관리 제외)
PROJECT_STATUS.md        단계별 실험 및 구현 기록
requirements.txt         애플리케이션 의존성(torch 제외)
```

`phase0/input`, `phase1/reference`, `phase3/test_audio`의 작은 WAV/MP3 fixture는 재현과 검증에 필요하므로 저장소에 포함합니다. 각 phase의 `output`, 진단용 생성 WAV, 로그, 가상환경과 모델 캐시는 로컬에 보존되지만 Git에는 포함하지 않습니다.

## RunPod Ubuntu/PyTorch 환경 구성

이 프로젝트는 로컬에서 Python 3.10.0으로 검증했습니다. RunPod에서는 Python 3.10을 권장하며, 먼저 선택한 PyTorch 이미지의 Python, CUDA, GPU 상태를 확인합니다.

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

저장소를 clone한 뒤 별도 가상환경을 만들 때는 시스템 site package를 노출하여 RunPod 이미지에 이미 설치된 CUDA 호환 PyTorch를 재사용합니다. 컨테이너가 작업별로 폐기되는 구성이라면 가상환경은 생략하고 이미지의 Python에 직접 설치해도 됩니다.

```bash
git clone <PRIVATE_GITHUB_REPOSITORY_URL> Speech-Caption-AI
cd Speech-Caption-AI
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt`는 로컬 프로젝트가 직접 사용하는 패키지만 명시하며 `torch`는 의도적으로 제외합니다. `pip install` 후에도 위의 Python 명령으로 `torch.cuda.is_available()`이 `True`인지 다시 확인하십시오. 이미지에 PyTorch가 없다면 RunPod 이미지의 CUDA 버전에 맞는 wheel을 PyTorch 공식 설치 안내에 따라 먼저 설치해야 합니다.

Ubuntu에서 `soundfile` 로딩 오류가 발생할 경우 시스템 라이브러리를 추가합니다.

```bash
apt-get update && apt-get install -y libsndfile1
```

### 모델 준비

모델 weight는 GitHub에 포함하지 않습니다. 인터넷 연결이 가능한 RunPod에서 최초 실행 시 다음처럼 다시 다운로드합니다.

```bash
source .venv/bin/activate
python - <<'PY'
from clearvoice import ClearVoice
from faster_whisper import WhisperModel

ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
    download_root="checkpoints/faster-whisper",
)
PY
```

ClearVoice의 `MossFormer2_SS_16K`와 faster-whisper `base`가 기본 모델입니다. Phase 3의 모델 크기 벤치마크는 `small`과 `medium`도 실행 시 다운로드합니다. SenseVoice/FunASR 경로를 실행하면 FunASR이 해당 모델을 별도로 내려받습니다. 다운로드된 파일은 `checkpoints/` 또는 각 라이브러리의 cache에 저장되며 Git에서 제외됩니다.

### RunPod 실행

저장소에 포함된 WAV로 오프라인 기본 파이프라인을 확인합니다.

```bash
source .venv/bin/activate
python phase0/run.py
```

GPU 실시간 파이프라인의 대표 실행 예시는 다음과 같습니다.

```bash
python phase3/run_overlap_pipeline.py \
  --duration 30 \
  --window-seconds 3 \
  --stride-seconds 2 \
  --assemble \
  --websocket \
  --json-output phase3/output/runpod_e2e.json
```

이 프로젝트 자체가 요구하는 API key, token 또는 필수 환경변수는 현재 없습니다. 비공개 모델을 추가로 사용할 경우 토큰은 `.env`나 RunPod Secret로 주입하고 저장소에는 커밋하지 마십시오.

RunPod의 일반적인 headless 컨테이너에는 Windows WASAPI/로컬 스피커 loopback 장치가 없습니다. `soundcard` 기반 실시간 캡처 단계는 Linux 오디오 장치와 PulseAudio/ALSA 구성이 별도로 있어야 하며, 장치가 없을 때는 포함된 WAV를 사용하는 오프라인 단계부터 검증하십시오. 웹 UI를 외부에서 열 경우 WebSocket 포트를 RunPod에서 노출하고, 현재 기본 bind 주소(`127.0.0.1`)가 사용 환경에 맞는지도 확인해야 합니다.
