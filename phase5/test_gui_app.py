"""Headless tests for the Windows GUI. No WASAPI or RunPod calls are made."""
from __future__ import annotations

import os
import threading
import time
import unittest
from typing import Union
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from phase5.gui_app import MainWindow
from phase5.subtitle_presentation import SubtitlePresentationState
from phase5.live_caption_controller import (
    EnvironmentInfo,
    LiveCaptionConfig,
    LiveCaptionController,
    LiveCaptionWorker,
    Phase4PipelineBackend,
)
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtCore import Qt
from typing_extensions import Self as TypingExtensionsSelf


class BlockingBackend:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.config = None

    def preflight(self, config: LiveCaptionConfig) -> EnvironmentInfo:
        self.config = config
        return EnvironmentInfo("테스트 스피커 (WASAPI loopback)", {"status": "ready"})

    def run(self, config: LiveCaptionConfig, hooks) -> int:
        self.started.set()
        hooks.on_subtitle(
            {
                "speaker": "speaker_0",
                "utterance_id": 1,
                "text": "테스트 자막",
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
        cls.app.setQuitOnLastWindowClosed(False)

    def pump_until(self, condition, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()
        self.assertTrue(condition(), "condition did not become true before timeout")

    @staticmethod
    def event(
        speaker: str,
        utterance_id: int,
        sequence: int,
        text: str,
        status: str = "partial",
    ) -> dict:
        return {
            "speaker": speaker,
            "utterance_id": utterance_id,
            "sequence": sequence,
            "status": status,
            "text": text,
            "timestamp": sequence * 1000,
        }

    @staticmethod
    def feed_texts(window: MainWindow) -> list[str]:
        return [
            window.subtitle_model.entry_at(row).text
            for row in range(window.subtitle_model.rowCount())
        ]

    def test_gui_import_and_main_window_creation(self) -> None:
        window = MainWindow()
        try:
            self.assertEqual(window.windowTitle(), "AI 실시간 자막")
            self.assertTrue(window.start_button.isEnabled())
            self.assertFalse(window.stop_button.isEnabled())
            self.assertEqual(window.speaker_mode_combo.currentData(), "single")
            self.assertEqual(window.speaker_mode_combo.currentText(), "일반 모드")
            overlap_item = window.speaker_mode_combo.model().item(1)
            self.assertFalse(overlap_item.isEnabled())
            self.assertEqual(overlap_item.toolTip(), "기능 개발 중입니다.")
            visible_labels = [label.text() for label in window.findChildren(QLabel)]
            self.assertFalse(any("Speaker" in text for text in visible_labels))
        finally:
            window.close()

    def test_idle_timer_expiry_updates_control_and_overlay_snapshot(self):
        window = MainWindow()
        now = [10.0]
        window.subtitle_model._presentation = SubtitlePresentationState(clock=lambda: now[0])
        overlay = window._ensure_overlay()
        snapshots = []
        window.subtitle_model.entries_changed.connect(snapshots.append)
        try:
            window.apply_subtitle_event(self.event("speaker_0", 1, 1, "확정", "final"))
            window.apply_subtitle_event(self.event("speaker_1", 1, 2, "진행"))
            self.assertTrue(window.subtitle_model._expiry_timer.isActive())
            now[0] = 14.0
            # Exercise the connected timer callback, without a new event/sleep.
            window.subtitle_model._expiry_timer.timeout.emit()
            self.assertEqual(self.feed_texts(window), ["진행"])
            self.assertEqual(overlay.entries, window.subtitle_model.entries)
            self.assertEqual(snapshots[-1], overlay.entries)
            window.apply_subtitle_event(self.event("speaker_1", 1, 3, "완료", "final"))
            now[0] = 18.0
            window.subtitle_model._expiry_timer.timeout.emit()
            self.assertEqual(window.subtitle_model.entries, ())
            self.assertEqual(overlay.entries, ())
            self.assertTrue(all(not label.text() for label in overlay._labels))
        finally:
            window.close()

    def test_stop_rejects_queued_old_worker_events_even_after_restart(self):
        backend = BlockingBackend()
        controller = LiveCaptionController(backend=backend)
        window = MainWindow(controller)
        try:
            window.start_captioning()
            self.pump_until(lambda: self.feed_texts(window) == ["테스트 자막"])
            old_worker = controller._worker
            window.apply_subtitle_event(self.event("speaker_1", 1, 2, "이전 확정", "final"))
            # Queue the signal in Qt BEFORE stop, then deliver it AFTER clear.
            from PySide6.QtCore import QMetaObject, Q_ARG
            QMetaObject.invokeMethod(old_worker, "subtitle_event", Qt.ConnectionType.QueuedConnection,
                                     Q_ARG("QVariantMap", self.event("speaker_0", 1, 9, "늦은 결과")))
            controller.stop()  # Also exercise stopping without the button wrapper.
            self.assertEqual(window.subtitle_model.entries, ())
            self.assertEqual(window.overlay.entries, ())
            self.pump_until(lambda: controller.state == "idle")
            self.assertEqual(window.subtitle_model.entries, ())
            window.start_captioning()
            self.pump_until(lambda: self.feed_texts(window) == ["테스트 자막"])
            # A separate obsolete worker simulates an old session delivery.
            obsolete = LiveCaptionWorker(LiveCaptionConfig(), backend=backend)
            obsolete.subtitle_event.connect(controller._forward_subtitle, Qt.ConnectionType.QueuedConnection)
            obsolete.subtitle_event.emit(self.event("speaker_0", 1, 99, "이전 세션"))
            self.app.processEvents()
            self.assertEqual(self.feed_texts(window), ["테스트 자막"])
        finally:
            if controller.state != "idle":
                window.stop_captioning()
                self.pump_until(lambda: controller.state == "idle")
            window.close()

    def test_pyside_python310_self_compatibility_is_primed(self) -> None:
        self.assertEqual(str(Union[int, TypingExtensionsSelf]).split("[")[0], "typing.Union")

    def test_other_stream_final_preserves_current_partial(self) -> None:
        window = MainWindow()
        try:
            window.apply_subtitle_event(
                self.event("speaker_0", 1, 10, "발표를 시작합니다.")
            )
            window.apply_subtitle_event(
                self.event("speaker_1", 1, 11, "네, 잘 들립니다.", "final")
            )
            self.assertEqual(self.feed_texts(window), ["네, 잘 들립니다.", "발표를 시작합니다."])
            self.assertEqual(
                window.subtitle_model.entry_at(0).speaker_id, "speaker_1"
            )
        finally:
            window.close()

    def test_two_stream_partials_share_the_snapshot(self) -> None:
        window = MainWindow()
        try:
            window.apply_subtitle_event(
                self.event("speaker_0", 4, 100, "나는 아직 저녁 안 먹었어")
            )
            window.apply_subtitle_event(
                self.event("speaker_1", 7, 101, "오늘 하루 어땠어?")
            )
            self.assertEqual(self.feed_texts(window), ["나는 아직 저녁 안 먹었어", "오늘 하루 어땠어?"])
        finally:
            window.close()

    def test_partial_updates_existing_entry_and_final_confirms_it(self) -> None:
        window = MainWindow()
        try:
            window.apply_subtitle_event(
                self.event("speaker_0", 2, 20, "오늘 저녁")
            )
            window.apply_subtitle_event(
                self.event("speaker_0", 2, 21, "오늘 저녁 아직 안 먹었어")
            )
            self.assertEqual(window.subtitle_model.rowCount(), 1)
            self.assertEqual(self.feed_texts(window), ["오늘 저녁 아직 안 먹었어"])
            self.assertEqual(window.subtitle_model.entry_at(0).status, "partial")

            window.apply_subtitle_event(
                self.event(
                    "speaker_0", 2, 22, "오늘 저녁 아직 안 먹었어", "final"
                )
            )
            self.assertEqual(window.subtitle_model.rowCount(), 1)
            self.assertEqual(window.subtitle_model.entry_at(0).status, "final")

            # A stale partial cannot reopen or duplicate a finalized utterance.
            window.apply_subtitle_event(
                self.event("speaker_0", 2, 23, "잘못된 후속 partial")
            )
            self.assertEqual(self.feed_texts(window), ["오늘 저녁 아직 안 먹었어"])
        finally:
            window.close()

    def test_new_utterance_appends_new_entry(self) -> None:
        window = MainWindow()
        try:
            window.apply_subtitle_event(
                self.event("speaker_1", 3, 30, "첫 번째 문장", "final")
            )
            window.apply_subtitle_event(
                self.event("speaker_1", 4, 31, "두 번째 문장")
            )
            self.assertEqual(self.feed_texts(window), ["첫 번째 문장", "두 번째 문장"])
        finally:
            window.close()

    def test_long_text_is_left_intact_for_qt_word_wrap(self) -> None:
        window = MainWindow()
        try:
            text = "이 문장은 UI 폭에 맞춰 자연스럽게 표시되며 데이터에 고정 폭 줄바꿈을 추가하지 않습니다."
            window.apply_subtitle_event(
                self.event("speaker_0", 1, 1, text)
            )
            self.assertEqual(self.feed_texts(window), [text])
            self.assertNotIn("\n", self.feed_texts(window)[0])
        finally:
            window.close()

    def test_feed_retains_only_recent_entries(self) -> None:
        window = MainWindow()
        try:
            for sequence in range(1, 91):
                window.apply_subtitle_event(
                    self.event(
                        f"speaker_{sequence % 2}",
                        sequence,
                        sequence,
                        f"문장 {sequence}",
                        "final",
                    )
                )
            self.assertEqual(window.subtitle_model.rowCount(), 2)
            self.assertEqual(window.subtitle_model.entry_at(0).text, "문장 89")
            self.assertEqual(window.subtitle_model.entry_at(1).text, "문장 90")
        finally:
            window.close()

    def test_overlay_lifecycle_reuses_one_window_and_snapshot(self) -> None:
        backend = BlockingBackend()
        controller = LiveCaptionController(backend=backend)
        window = MainWindow(controller=controller)
        try:
            window.start_captioning()
            self.pump_until(backend.started.is_set)
            self.pump_until(
                lambda: window.overlay is not None
                and window.overlay.isVisible()
                and [entry.text for entry in window.overlay.entries] == ["테스트 자막"]
            )
            overlay = window.overlay
            self.assertIsNotNone(overlay)
            self.assertTrue(overlay.style_config.bold)

            window.stop_captioning()
            self.assertFalse(overlay.isVisible())
            self.assertEqual(overlay.entries, ())
            self.pump_until(lambda: controller.state == "idle")

            window.start_captioning()
            self.pump_until(lambda: overlay.isVisible())
            self.assertIs(window.overlay, overlay)
        finally:
            if controller.state != "idle":
                window.stop_captioning()
                self.pump_until(lambda: controller.state == "idle")
            window.close()

    def test_overlay_settings_apply_without_recreation(self) -> None:
        window = MainWindow()
        try:
            overlay = window._ensure_overlay()
            window.overlay_font_size.setValue(34)
            window.overlay_opacity.setValue(55)
            window.overlay_bold_checkbox.setChecked(False)
            self.assertIs(window._ensure_overlay(), overlay)
            self.assertEqual(overlay.style_config.font_size, 34)
            self.assertEqual(overlay.style_config.background_opacity, 55)
            self.assertFalse(overlay.style_config.bold)
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
                lambda: self.feed_texts(window) == ["테스트 자막"]
            )
            self.assertEqual(window.latency_label.text(), "최근 latency: 125 ms")
            controller.stop()
            self.pump_until(lambda: controller.state == "idle")
        finally:
            if controller.state != "idle":
                controller.stop()
                self.pump_until(lambda: controller.state == "idle")
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
        self.assertEqual(argv[argv.index("--speaker-mode") + 1], "single")
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
