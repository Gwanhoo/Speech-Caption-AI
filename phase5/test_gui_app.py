"""Headless tests for the Windows GUI. No WASAPI or RunPod calls are made."""
from __future__ import annotations

import os
import threading
import time
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from phase5.gui_app import MainWindow
from phase5.live_caption_controller import (
    EnvironmentInfo,
    LiveCaptionConfig,
    LiveCaptionController,
    LiveCaptionWorker,
    Phase4PipelineBackend,
)


class BlockingBackend:
    def __init__(self) -> None:
        self.started = threading.Event()

    def preflight(self, config: LiveCaptionConfig) -> EnvironmentInfo:
        return EnvironmentInfo("테스트 스피커 (WASAPI loopback)", {"status": "ready"})

    def run(self, config: LiveCaptionConfig, hooks) -> int:
        self.started.set()
        hooks.on_subtitle(
            {
                "speaker": "speaker_0",
                "text": "테스트 자막",
                "assembled_text": "테스트 자막",
                "status": "partial",
                "sequence": 1,
                "timestamp": 1,
            }
        )
        hooks.on_metric({"post_capture_latency_seconds": 0.125})
        hooks.stop_event.wait(2)
        return 0


class ExplodingBackend:
    def preflight(self, config: LiveCaptionConfig) -> EnvironmentInfo:
        raise RuntimeError("WASAPI capture 실패")

    def run(self, config: LiveCaptionConfig, hooks) -> int:
        raise AssertionError("preflight failure must prevent pipeline start")


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def pump_until(self, condition, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()
        self.assertTrue(condition(), "condition did not become true before timeout")

    def test_gui_import_and_main_window_creation(self) -> None:
        window = MainWindow()
        try:
            self.assertEqual(window.windowTitle(), "AI 실시간 자막")
            self.assertTrue(window.start_button.isEnabled())
            self.assertFalse(window.stop_button.isEnabled())
        finally:
            window.close()

    def test_subtitle_event_updates_speaker_a_and_b(self) -> None:
        window = MainWindow()
        try:
            window.apply_subtitle_event(
                {
                    "speaker": "speaker_0",
                    "assembled_text": "발표를 시작합니다.",
                    "status": "partial",
                    "sequence": 10,
                }
            )
            window.apply_subtitle_event(
                {
                    "speaker": "speaker_1",
                    "assembled_text": "네, 잘 들립니다.",
                    "status": "final",
                    "sequence": 11,
                }
            )
            self.assertEqual(window.speaker_a.text.toPlainText(), "발표를 시작합니다.")
            self.assertEqual(window.speaker_b.text.toPlainText(), "네, 잘 들립니다.")
            self.assertEqual(window.speaker_a.event_status.text(), "인식 중")
            self.assertEqual(window.speaker_b.event_status.text(), "확정")
        finally:
            window.close()

    def test_duplicate_start_and_graceful_stop_state_transition(self) -> None:
        backend = BlockingBackend()
        controller = LiveCaptionController(backend=backend)
        states: list[str] = []
        controller.state_changed.connect(states.append)

        self.assertTrue(controller.start(LiveCaptionConfig(websocket_enabled=False)))
        self.assertFalse(controller.start(LiveCaptionConfig(websocket_enabled=False)))
        self.pump_until(backend.started.is_set)
        self.assertIn(controller.state, {"running", "starting"})
        self.assertTrue(controller.stop())
        self.assertEqual(controller.state, "stopping")
        self.pump_until(lambda: controller.state == "idle")
        self.assertIn("starting", states)
        self.assertIn("stopping", states)
        self.assertEqual(states[-1], "idle")

    def test_structured_worker_event_reaches_window(self) -> None:
        backend = BlockingBackend()
        controller = LiveCaptionController(backend=backend)
        window = MainWindow(controller=controller)
        try:
            self.assertTrue(controller.start(LiveCaptionConfig(websocket_enabled=False)))
            self.pump_until(
                lambda: window.speaker_a.text.toPlainText() == "테스트 자막"
            )
            self.assertEqual(window.latency_label.text(), "최근 latency: 125 ms")
            controller.stop()
            self.pump_until(lambda: controller.state == "idle")
        finally:
            window.close()

    def test_worker_exception_becomes_gui_error_state(self) -> None:
        controller = LiveCaptionController(backend=ExplodingBackend())
        window = MainWindow(controller=controller)
        try:
            self.assertTrue(controller.start(LiveCaptionConfig(websocket_enabled=False)))
            self.pump_until(lambda: controller.state == "idle")
            self.assertIn("오류:", window.status_label.text())
            self.assertIn("WASAPI capture 실패", window.status_label.text())
            self.assertIn("연결 안 됨", window.server_state_label.text())
        finally:
            window.close()

    def test_backend_reuses_phase4_pipeline_with_fixed_live_settings(self) -> None:
        backend = Phase4PipelineBackend()
        config = LiveCaptionConfig(server_url="https://example.invalid")
        hooks = object()
        pipeline_module = Mock()
        pipeline_module.main.return_value = 0
        with patch(
            "phase5.live_caption_controller._load_phase4_pipeline",
            return_value=pipeline_module,
        ):
            self.assertEqual(backend.run(config, hooks), 0)
        main = pipeline_module.main
        argv = main.call_args.args[0]
        self.assertIn("--live", argv)
        self.assertEqual(argv[argv.index("--processing-mode") + 1], "remote")
        self.assertEqual(argv[argv.index("--window-seconds") + 1], "3")
        self.assertEqual(argv[argv.index("--stride-seconds") + 1], "2")
        self.assertIn("--assemble", argv)
        self.assertIn("--websocket", argv)
        self.assertIs(main.call_args.kwargs["runtime_hooks"], hooks)

    def test_worker_can_be_constructed(self) -> None:
        worker = LiveCaptionWorker(
            LiveCaptionConfig(websocket_enabled=False), backend=BlockingBackend()
        )
        self.assertFalse(worker.stop_event.is_set())
        worker.request_stop()
        self.assertTrue(worker.stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
