from __future__ import annotations

import sys
import os
import threading
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Protocol

# PySide6 6.8.x adds an incompatible ``typing.Self`` on Python 3.10.
# Resolve the backport first so later PyTorch imports keep the valid object.
from typing_extensions import Self as _TypingExtensionsSelf  # noqa: F401

from PySide6.QtCore import QObject, QThread, Signal, Slot


ROOT = Path(__file__).resolve().parent.parent
for module_directory in (ROOT / "phase3", ROOT / "phase4"):
    path = str(module_directory)
    if path not in sys.path:
        sys.path.insert(0, path)

from remote_gpu_client import (  # noqa: E402
    RemoteConnectionError,
    RemoteGPUClient,
    RemoteHTTPError,
    RemoteProtocolError,
    RemoteTimeoutError,
)


# Keep the deployment address in one place.  The GUI exposes this value for a
# per-run override, while installations set GPU_SERVER_URL before launch.
DEFAULT_REMOTE_SERVER_URL = os.environ.get(
    "GPU_SERVER_URL", "https://1djm8of460wbeh-8787.proxy.runpod.net"
).rstrip("/")
DEFAULT_GUI_OUTPUT = ROOT / "phase5" / "output" / "gui_last_run.json"


@dataclass(frozen=True)
class LiveCaptionConfig:
    server_url: str = DEFAULT_REMOTE_SERVER_URL
    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    websocket_enabled: bool = True
    websocket_host: str = "127.0.0.1"
    websocket_port: int = 8765
    json_output: Path = DEFAULT_GUI_OUTPUT


@dataclass(frozen=True)
class EnvironmentInfo:
    device_name: str
    health: dict[str, Any]


@dataclass
class RuntimeHooks:
    """Duck-typed equivalent of Phase 4's optional frontend hooks."""

    stop_event: threading.Event = field(default_factory=threading.Event)
    on_subtitle: Callable[[dict[str, Any]], None] | None = None
    on_metric: Callable[[dict[str, Any]], None] | None = None


class PipelineBackend(Protocol):
    def preflight(self, config: LiveCaptionConfig) -> EnvironmentInfo: ...

    def run(
        self,
        config: LiveCaptionConfig,
        hooks: RuntimeHooks,
    ) -> int: ...


def _load_phase4_pipeline() -> Any:
    # run_overlap_pipeline imports soundcard, whose Linux backend connects to
    # PulseAudio during import.  Delay it so headless GUI imports stay safe.
    return import_module("run_overlap_pipeline")


class Phase4PipelineBackend:
    """Thin adapter around the existing Phase 4-I orchestration."""

    def preflight(self, config: LiveCaptionConfig) -> EnvironmentInfo:
        client = RemoteGPUClient(
            config.server_url,
            connect_timeout=config.connect_timeout,
            read_timeout=config.read_timeout,
        )
        try:
            health = client.health()
        finally:
            client.close()

        if sys.platform != "win32":
            raise RuntimeError("Windows WASAPI 환경에서만 시스템 오디오를 캡처할 수 있습니다.")
        sc = import_module("soundcard")
        speaker = sc.default_speaker()
        if speaker is None:
            raise RuntimeError("기본 Windows 출력 오디오 장치를 찾을 수 없습니다.")
        loopback = sc.get_microphone(speaker.name, include_loopback=True)
        if loopback is None:
            raise RuntimeError("기본 출력 장치의 WASAPI loopback을 찾을 수 없습니다.")
        device_name = getattr(loopback, "name", None) or speaker.name
        return EnvironmentInfo(
            device_name=f"{device_name} (WASAPI loopback)",
            health=health,
        )

    def run(
        self,
        config: LiveCaptionConfig,
        hooks: RuntimeHooks,
    ) -> int:
        pipeline = _load_phase4_pipeline()
        argv = [
            "--live",
            "--processing-mode",
            "remote",
            "--remote-server-url",
            config.server_url,
            "--remote-connect-timeout",
            str(config.connect_timeout),
            "--remote-read-timeout",
            str(config.read_timeout),
            "--window-seconds",
            "3",
            "--stride-seconds",
            "2",
            "--assemble",
            "--json-output",
            str(config.json_output),
        ]
        if config.websocket_enabled:
            argv.extend(
                [
                    "--websocket",
                    "--ws-host",
                    config.websocket_host,
                    "--ws-port",
                    str(config.websocket_port),
                ]
            )
        return pipeline.main(argv, runtime_hooks=hooks)


def user_error_message(exc: BaseException) -> str:
    detail = str(exc).strip()
    if isinstance(exc, RemoteTimeoutError):
        return f"RunPod 서버 응답 시간이 초과되었습니다. {detail}".strip()
    if isinstance(exc, RemoteConnectionError):
        return f"RunPod 서버에 연결할 수 없습니다. {detail}".strip()
    if isinstance(exc, RemoteHTTPError):
        return f"RunPod 서버가 오류를 반환했습니다. {detail}".strip()
    if isinstance(exc, RemoteProtocolError):
        return f"RunPod 서버 응답 형식이 올바르지 않습니다. {detail}".strip()
    if detail:
        return detail
    return type(exc).__name__


class LiveCaptionWorker(QObject):
    server_state = Signal(bool, str)
    device_state = Signal(str)
    status = Signal(str)
    subtitle_event = Signal(dict)
    latency_updated = Signal(float)
    error = Signal(str)
    finished = Signal(bool, str)

    def __init__(
        self,
        config: LiveCaptionConfig,
        backend: PipelineBackend | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.backend = backend or Phase4PipelineBackend()
        self.stop_event = threading.Event()

    def request_stop(self) -> None:
        """Thread-safe; capture observes this event between recorder reads."""
        self.stop_event.set()

    @Slot()
    def run(self) -> None:
        try:
            self.status.emit("RunPod 서버 상태 확인 중…")
            info = self.backend.preflight(self.config)
            self.server_state.emit(True, "연결됨")
            self.device_state.emit(info.device_name)
            if self.stop_event.is_set():
                self.status.emit("대기 중")
                self.finished.emit(True, "시작이 취소되었습니다.")
                return

            self.status.emit("오디오 캡처 및 원격 자막 처리 중…")
            hooks = RuntimeHooks(
                stop_event=self.stop_event,
                on_subtitle=self._publish_subtitle,
                on_metric=self._publish_metric,
            )
            exit_code = self.backend.run(self.config, hooks)
            if exit_code != 0:
                message = (
                    "파이프라인이 일부 창을 처리하지 못했습니다. "
                    f"결과 로그: {self.config.json_output}"
                )
                self.error.emit(message)
                self.finished.emit(False, message)
                return
            message = "자막 처리가 안전하게 종료되었습니다."
            self.status.emit("대기 중")
            self.finished.emit(True, message)
        except BaseException as exc:
            message = user_error_message(exc)
            self.server_state.emit(False, "연결 안 됨")
            self.error.emit(message)
            self.finished.emit(False, message)

    def _publish_subtitle(self, event: dict[str, Any]) -> None:
        if not self.stop_event.is_set():
            self.subtitle_event.emit(dict(event))

    def _publish_metric(self, metric: dict[str, Any]) -> None:
        latency = metric.get("post_capture_latency_seconds")
        if isinstance(latency, (int, float)):
            self.latency_updated.emit(float(latency) * 1000)


class LiveCaptionController(QObject):
    server_state = Signal(bool, str)
    device_state = Signal(str)
    status = Signal(str)
    subtitle_event = Signal(dict)
    latency_updated = Signal(float)
    error = Signal(str)
    state_changed = Signal(str)
    run_finished = Signal(bool, str)

    def __init__(self, backend: PipelineBackend | None = None) -> None:
        super().__init__()
        self._backend = backend
        self._state = "idle"
        self._thread: QThread | None = None
        self._worker: LiveCaptionWorker | None = None
        self._pending_result: tuple[bool, str] = (True, "")

    @property
    def state(self) -> str:
        return self._state

    def start(self, config: LiveCaptionConfig) -> bool:
        if self._state != "idle":
            self.status.emit("이미 자막 파이프라인이 실행 중입니다.")
            return False

        self._set_state("starting")
        thread = QThread(self)
        worker = LiveCaptionWorker(config, backend=self._backend)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.server_state.connect(self.server_state)
        worker.device_state.connect(self.device_state)
        worker.status.connect(self.status)
        worker.subtitle_event.connect(self._forward_subtitle)
        worker.latency_updated.connect(self.latency_updated)
        worker.error.connect(self.error)
        worker.device_state.connect(self._mark_running)
        worker.finished.connect(self._remember_result)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)

        self._thread = thread
        self._worker = worker
        thread.start()
        return True

    def stop(self) -> bool:
        if self._state not in {"starting", "running"} or self._worker is None:
            return False
        self._set_state("stopping")
        self.status.emit("중지 요청됨 — 현재 작업을 마치고 안전하게 종료 중…")
        self._worker.request_stop()
        return True

    @Slot(dict)
    def _forward_subtitle(self, event: dict[str, Any]) -> None:
        # This slot runs on the GUI thread. A worker-side stop check alone
        # cannot invalidate Qt signals already queued before STOP/restart.
        if self._state in {"starting", "running"} and self.sender() is self._worker:
            self.subtitle_event.emit(event)

    @Slot(str)
    def _mark_running(self, _device_name: str) -> None:
        if self._state == "starting":
            self._set_state("running")

    @Slot(bool, str)
    def _remember_result(self, success: bool, message: str) -> None:
        self._pending_result = (success, message)

    @Slot()
    def _thread_finished(self) -> None:
        result = self._pending_result
        thread = self._thread
        self._worker = None
        self._thread = None
        self._set_state("idle")
        self.run_finished.emit(*result)
        if thread is not None:
            thread.deleteLater()

    def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        self.state_changed.emit(state)
