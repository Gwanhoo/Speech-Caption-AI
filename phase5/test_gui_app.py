"""Headless tests for the Windows GUI. No WASAPI or RunPod calls are made."""
from __future__ import annotations

import gc
import os
import logging
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Union
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from phase5.gui_app import MainWindow, main as gui_main, parse_launch_options
from phase5.subtitle_presentation import SubtitleFeedEntry, SubtitlePresentationState
from phase5.replay_overlay_timeline import DEFAULT_TIMELINE, TimelineReplay, load_events
from phase5.live_caption_controller import (
    EnvironmentInfo,
    LOGGER,
    LiveCaptionConfig,
    LiveCaptionController,
    LiveCaptionWorker,
    Phase4PipelineBackend,
    SpeakerMode,
    configure_gui_runtime,
    shutdown_gui_runtime,
    user_error_message,
)
from remote_gpu_client import RemoteConnectionError, RemoteTimeoutError
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QColor
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


class SilentBackend(BlockingBackend):
    """Hold preflight and produce no STT events, without any network calls."""

    def __init__(self) -> None:
        super().__init__()
        self.release_preflight = threading.Event()

    def preflight(self, config: LiveCaptionConfig) -> EnvironmentInfo:
        if not self.release_preflight.wait(10):
            raise RuntimeError("test preflight was not released")
        return super().preflight(config)

    def run(self, config: LiveCaptionConfig, hooks) -> int:
        self.started.set()
        hooks.stop_event.wait(10)
        return 0


class RecoveringBackend(BlockingBackend):
    """First health check fails; a later user-initiated start succeeds."""

    def __init__(self) -> None:
        super().__init__()
        self.preflight_attempts = 0

    def preflight(self, config: LiveCaptionConfig) -> EnvironmentInfo:
        self.preflight_attempts += 1
        if self.preflight_attempts == 1:
            raise RemoteConnectionError("mock server unavailable")
        return super().preflight(config)


class GuiTests(unittest.TestCase):
    def setUp(self) -> None:
        # Device enumeration must not open WASAPI/PulseAudio in headless tests.
        speaker = SimpleNamespace(id="{test-default}", name="Test output", channels=2)
        soundcard = Mock()
        soundcard.default_speaker.return_value = speaker
        soundcard.all_speakers.return_value = [speaker]
        self.soundcard_patch = patch.dict(sys.modules, {"soundcard": soundcard})
        self.soundcard_patch.start()
        self.addCleanup(self.soundcard_patch.stop)

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

    def tearDown(self) -> None:
        # Headless tests do not run QApplication.exec(), so Qt's deferred
        # deletes must be drained explicitly.  Collect Python wrapper cycles
        # here on the GUI thread instead of letting a later worker thread do it.
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        gc.collect()

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

    def test_recorded_window_13_14_final_and_partial_keep_display_order(self) -> None:
        events = load_events(DEFAULT_TIMELINE)
        original = load_events(DEFAULT_TIMELINE)
        replay = TimelineReplay(events)
        replay.overlay.show()
        expected = {
            11: [(2, "final")],
            12: [(2, "final"), (3, "partial")],
            13: [(2, "final"), (3, "final")],
            14: [(2, "final"), (3, "final"), (4, "partial")],
        }
        try:
            for index, event in enumerate(events[:14]):
                replay.advance(replay.offsets[index], through_sequence=event["sequence"])
                self.app.processEvents()
                if event["sequence"] in expected:
                    self.assertEqual(
                        [(entry.utterance_id, entry.status) for entry in replay.model.entries],
                        expected[event["sequence"]],
                    )
                    self.assertEqual(replay.overlay.entries, replay.model.entries)
                    for entry, label in zip(replay.overlay.entries, replay.overlay._labels):
                        self.assertEqual(label.text(), entry.text)
                        self.assertGreaterEqual(label.height(), label.heightForWidth(label.width()))
            self.assertEqual(replay.accepted_sequences, list(range(1, 15)))
            self.assertEqual(events, original)  # Wire timestamps/text/order untouched.
            self.assertAlmostEqual(replay.now, 19.469)
        finally:
            replay.close()

    def test_recorded_final_expiry_uses_arrival_intervals_not_stream_end(self) -> None:
        replay = TimelineReplay(load_events(DEFAULT_TIMELINE))
        try:
            replay.advance(19.469)
            replay.advance(22.199)
            self.assertEqual([e.utterance_id for e in replay.model.entries], [2, 3, 4])
            replay.advance(22.201)
            self.assertEqual([e.utterance_id for e in replay.model.entries], [3, 4])
            # At the next recorded pair, the previous FINAL has only 95ms left.
            replay.advance(23.373)
            self.assertEqual([(e.utterance_id, e.status) for e in replay.model.entries],
                             [(3, "final"), (4, "final"), (5, "partial")])
            replay.advance(23.469)
            self.assertEqual([e.utterance_id for e in replay.model.entries], [4, 5])
            replay.advance(31.895)
            self.assertEqual([e.utterance_id for e in replay.model.entries], [6])
            replay.advance(31.897)
            self.assertEqual(replay.model.entries, ())
            self.assertEqual(replay.overlay.entries, ())
            self.assertTrue(replay.overlay._card.isHidden())
            self.assertEqual(replay.overlay.height(), replay.overlay.EMPTY_HEIGHT)
            self.assertEqual(replay.accepted_sequences, list(range(1, 22)))
        finally:
            replay.close()

    def test_recorded_final_duplicate_does_not_renew_deadline_or_reappear(self) -> None:
        replay = TimelineReplay(load_events(DEFAULT_TIMELINE))
        final = replay.events[10]  # Recorded window 13 FINAL, sequence 11.
        try:
            replay.advance(19.469)
            self.assertFalse(replay.model.upsert_event(dict(final)))
            # Synthetic duplicate delivery with a newer sequence is also invalid.
            self.assertFalse(replay.model.upsert_event(dict(final, sequence=100)))
            replay.advance(22.201)
            self.assertNotIn(2, [e.utterance_id for e in replay.model.entries])
            self.assertFalse(replay.model.upsert_event(dict(final, sequence=101)))
            replay.advance(31.897)
            self.assertEqual(replay.accepted_sequences, list(range(1, 22)))
            self.assertEqual(replay.model.entries, ())
        finally:
            replay.close()

    def test_real_qt_timer_removes_expired_recorded_final_without_new_event(self) -> None:
        window = MainWindow()
        now = [0.0]
        window.subtitle_model._presentation = SubtitlePresentationState(clock=lambda: now[0])
        overlay = window._ensure_overlay()
        try:
            window.apply_subtitle_event(load_events(DEFAULT_TIMELINE)[10])
            self.assertEqual(len(overlay.entries), 1)
            now[0] = 4.001
            # Process the real 100ms Qt timer, rather than emitting timeout by hand.
            self.pump_until(lambda: not overlay.entries)
            self.assertEqual(window.subtitle_model.entries, ())
            self.assertTrue(overlay._card.isHidden())
        finally:
            window.close()

    def test_recorded_overlay_flags_and_initial_geometry(self) -> None:
        replay = TimelineReplay(load_events(DEFAULT_TIMELINE))
        overlay = replay.overlay
        try:
            replay.advance(19.469)
            overlay.show()
            self.app.processEvents()
            available = self.app.primaryScreen().availableGeometry()
            self.assertTrue(available.contains(overlay.geometry()))
            self.assertTrue(overlay.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
            self.assertTrue(overlay.windowFlags() & Qt.WindowType.FramelessWindowHint)
            self.assertTrue(overlay.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground))
            self.assertTrue(overlay.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating))
            self.assertEqual(overlay.style_config.background_opacity, 72)
            self.assertIn("rgba(10, 15, 25, 184)", overlay._card.styleSheet())
        finally:
            replay.close()

    def test_overlay_regrows_after_empty_and_preserves_bottom_or_drag_anchor(self) -> None:
        window = MainWindow()
        overlay = window._ensure_overlay()
        entries = tuple(SubtitleFeedEntry("speaker_0", i, i, "partial", f"자막 {i}")
                        for i in range(1, 4))
        try:
            overlay.show()
            self.app.processEvents()
            bottom = overlay.geometry().bottom()
            for count in (3, 2, 1, 0, 1, 2, 3):
                overlay.set_entries(entries[:count])
                self.app.processEvents()
                self.assertEqual(overlay.height(), overlay.layout_metrics.required_height)
                self.assertEqual(overlay.geometry().bottom(), bottom)
            # Simulate the state established by dragging; resizing must keep it.
            overlay.move(100, 100)
            overlay._was_dragged = True
            for count in (1, 0, 3):
                overlay.set_entries(entries[:count])
                self.app.processEvents()
                self.assertEqual(overlay.pos().x(), 100)
                self.assertEqual(overlay.pos().y(), 100)
                self.assertEqual(overlay.height(), overlay.layout_metrics.required_height)
        finally:
            window.close()

    def test_gui_import_and_main_window_creation(self) -> None:
        window = MainWindow()
        try:
            self.assertEqual(window.windowTitle(), "AI 실시간 자막")
            self.assertTrue(window.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose))
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

    def test_selected_output_device_id_is_passed_from_gui_to_backend(self) -> None:
        backend = BlockingBackend()
        controller = LiveCaptionController(backend=backend)
        window = MainWindow(controller)
        try:
            window.output_device_combo.clear()
            window.output_device_combo.addItem("Windows 기본 출력 장치", None)
            window.output_device_combo.addItem("동일 이름 장치 (id-a)", "{id-a}")
            window.output_device_combo.addItem("동일 이름 장치 (id-b)", "{id-b}")
            window.output_device_combo.setCurrentIndex(2)
            window.start_captioning()
            self.pump_until(backend.started.is_set)
            self.assertEqual(backend.config.output_device_id, "{id-b}")
        finally:
            if controller.state != "idle":
                window.stop_captioning()
                self.pump_until(lambda: controller.state == "idle")
            window.close()

    def test_output_device_refresh_preserves_surviving_selection(self) -> None:
        devices = [
            SimpleNamespace(device_id="{id-a}", name="같은 이름", is_default=True),
            SimpleNamespace(device_id="{id-b}", name="같은 이름", is_default=False),
        ]
        with patch("phase5.gui_app.list_output_devices", return_value=devices):
            window = MainWindow()
            try:
                window.output_device_combo.setCurrentIndex(
                    window.output_device_combo.findData("{id-b}")
                )
                window.refresh_output_devices()
                self.assertEqual(window.output_device_combo.currentData(), "{id-b}")
                self.assertIn("유지", window.device_label.text())
            finally:
                window.close()

    def test_output_device_refresh_reverts_missing_selection_to_default(self) -> None:
        initial_devices = [
            SimpleNamespace(device_id="{id-a}", name="스피커", is_default=True),
            SimpleNamespace(device_id="{id-b}", name="헤드셋", is_default=False),
        ]
        remaining_devices = [
            SimpleNamespace(device_id="{id-a}", name="스피커", is_default=True),
        ]
        with patch(
            "phase5.gui_app.list_output_devices",
            side_effect=[initial_devices, remaining_devices],
        ):
            window = MainWindow()
            try:
                window.output_device_combo.setCurrentIndex(
                    window.output_device_combo.findData("{id-b}")
                )
                window.refresh_output_devices()
                self.assertIsNone(window.output_device_combo.currentData())
                self.assertIn("사라져", window.device_label.text())
            finally:
                window.close()

    def test_no_output_device_blocks_start_and_keeps_refresh_available(self) -> None:
        backend = BlockingBackend()
        with patch("phase5.gui_app.list_output_devices", return_value=[]):
            controller = LiveCaptionController(backend=backend)
            window = MainWindow(controller)
            try:
                self.assertFalse(window.start_button.isEnabled())
                self.assertTrue(window.refresh_device_button.isEnabled())
                window.start_captioning()
                self.assertEqual(controller.state, "idle")
                self.assertIsNone(backend.config)
                self.assertIn("사용 가능한 Windows 출력 장치가 없습니다", window.status_label.text())
            finally:
                window.close()

    def test_output_device_enumeration_failure_blocks_start_with_error(self) -> None:
        backend = BlockingBackend()
        with patch(
            "phase5.gui_app.list_output_devices",
            side_effect=RuntimeError("WASAPI enumeration failed"),
        ):
            controller = LiveCaptionController(backend=backend)
            window = MainWindow(controller)
            try:
                self.assertFalse(window.start_button.isEnabled())
                window.start_captioning()
                self.assertEqual(controller.state, "idle")
                self.assertIsNone(backend.config)
                self.assertIn("오디오 장치 목록을 가져올 수 없습니다", window.status_label.text())
            finally:
                window.close()

    def test_refresh_enumeration_failure_disables_start_after_prior_success(self) -> None:
        devices = [
            SimpleNamespace(device_id="{id-a}", name="스피커", is_default=True),
        ]
        with patch(
            "phase5.gui_app.list_output_devices",
            side_effect=[devices, RuntimeError("WASAPI enumeration failed")],
        ):
            window = MainWindow()
            try:
                self.assertTrue(window.start_button.isEnabled())
                window.refresh_output_devices()
                self.assertFalse(window.start_button.isEnabled())
                self.assertTrue(window.refresh_device_button.isEnabled())
                self.assertIn("가져올 수 없습니다", window.device_label.text())
            finally:
                window.close()

    def test_device_refresh_controls_are_disabled_while_running(self) -> None:
        window = MainWindow()
        try:
            window.apply_controller_state("running")
            self.assertFalse(window.output_device_combo.isEnabled())
            self.assertFalse(window.refresh_device_button.isEnabled())
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

    def test_empty_overlay_is_visible_before_preflight_and_reused_after_restart(self) -> None:
        backend = SilentBackend()
        controller = LiveCaptionController(backend=backend)
        window = MainWindow(controller)
        window.overlay_font_size.setValue(34)
        window.overlay_opacity.setValue(20)
        window.overlay_bold_checkbox.setChecked(False)
        window.overlay_text_color_combo.setCurrentIndex(1)
        now = [0.0]
        window.subtitle_model._presentation = SubtitlePresentationState(clock=lambda: now[0])
        overlay = None
        try:
            for _ in range(2):
                backend.release_preflight.clear()
                window.start_button.click()
                # Assert synchronously: neither preflight nor an STT result can finish.
                self.assertEqual(controller.state, "starting")
                if overlay is None:
                    overlay = window.overlay
                self.assertIs(window.overlay, overlay)
                self.assertTrue(overlay.isVisible())
                self.assertTrue(overlay._card.isVisible())
                self.assertEqual(overlay.entries, ())
                self.assertTrue(all(label.text() == "" for label in overlay._labels))
                self.assertEqual(overlay.height(), overlay._font_line_height() + overlay._non_text_height(1))
                self.assertEqual(overlay.style_config, window._overlay_style())
                self.assertIn("rgba(10, 15, 25, 51)", overlay._card.styleSheet())
                self.assertEqual(window.subtitle_model.receivers("2entries_changed(PyObject)"), 1)
                self.app.processEvents()
                self.assertTrue(overlay._card.isVisible())
                backend.release_preflight.set()
                self.pump_until(lambda: controller.state == "running")
                self.assertEqual(overlay.entries, ())
                overlay.move(100, 100)
                overlay._was_dragged = True
                position = overlay.pos()
                with patch.object(overlay, "show", wraps=overlay.show) as show:
                    for sequence, status in ((1, "partial"), (2, "final")):
                        controller._worker.subtitle_event.emit(self.event("speaker_0", 1, sequence, "첫 자막", status))
                        self.pump_until(lambda: bool(overlay.entries) and overlay.entries[0].status == status)
                        self.assertEqual(overlay._labels[0].text(), "첫 자막")
                        self.assertEqual(overlay.pos(), position)
                        self.assertEqual(overlay.style_config, window._overlay_style())
                    show.assert_not_called()
                now[0] += 5
                window.subtitle_model.expire_final_entries()
                self.assertEqual(overlay.entries, ())
                self.assertTrue(overlay._card.isVisible())
                window.stop_button.click()
                self.assertFalse(overlay.isVisible())
                self.assertEqual(overlay.entries, ())
                self.pump_until(lambda: controller.state == "idle")
        finally:
            backend.release_preflight.set()
            if controller.state != "idle":
                window.stop_captioning()
                self.pump_until(lambda: controller.state == "idle")
            window.close()

    def test_empty_overlay_respects_visibility_toggle_during_session(self) -> None:
        backend = SilentBackend()
        controller = LiveCaptionController(backend=backend)
        window = MainWindow(controller)
        try:
            window.overlay_visible_checkbox.setChecked(False)
            window.start_button.click()
            overlay = window.overlay
            self.assertFalse(overlay.isVisible())
            window.overlay_visible_checkbox.setChecked(True)
            self.assertTrue(overlay.isVisible())
            self.assertTrue(overlay._card.isVisible())
            self.assertEqual(overlay.entries, ())
            window.overlay_visible_checkbox.setChecked(False)
            self.assertFalse(overlay.isVisible())
            window.stop_button.click()
            window.overlay_visible_checkbox.setChecked(True)
            self.assertFalse(overlay.isVisible())
            self.assertIs(window.overlay, overlay)
        finally:
            backend.release_preflight.set()
            if controller.state != "idle":
                window.stop_captioning()
                self.pump_until(lambda: controller.state == "idle")
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
            window.overlay_text_color_combo.setCurrentIndex(
                window.overlay_text_color_combo.findData("#FDE047")
            )
            self.assertIs(window._ensure_overlay(), overlay)
            self.assertEqual(overlay.style_config.font_size, 34)
            self.assertEqual(overlay.style_config.background_opacity, 55)
            self.assertFalse(overlay.style_config.bold)
            self.assertEqual(overlay.style_config.text_color, "#FDE047")
        finally:
            window.close()

    def test_custom_overlay_color_updates_visible_partial_and_final_without_recreation(self) -> None:
        window = MainWindow()
        overlay = window._ensure_overlay()
        try:
            window.overlay_font_size.setValue(34)
            window.overlay_opacity.setValue(55)
            window.overlay_bold_checkbox.setChecked(False)
            window.apply_subtitle_event(
                self.event("speaker_0", 1, 1, "확정 자막", "final")
            )
            window.apply_subtitle_event(self.event("speaker_0", 2, 2, "진행 중 자막"))

            with patch(
                "phase5.gui_app.QColorDialog.getColor", return_value=QColor("#1234ab")
            ) as get_color:
                window.overlay_custom_color_button.click()

            self.assertIs(window.overlay, overlay)
            self.assertEqual(window.overlay_text_color_combo.currentText(), "사용자 지정 색상")
            self.assertEqual(window.overlay_text_color_combo.currentData(), "#1234AB")
            self.assertEqual(overlay.style_config.text_color, "#1234AB")
            self.assertEqual(overlay.style_config.font_size, 34)
            self.assertEqual(overlay.style_config.background_opacity, 55)
            self.assertFalse(overlay.style_config.bold)
            self.assertEqual(get_color.call_args.args[0].name().upper(), "#FFFFFF")
            self.assertEqual(get_color.call_args.args[2], "자막 글자 색상 선택")

            final_label, partial_label = overlay._labels[:2]
            self.assertIn("18, 52, 171, 255", final_label.styleSheet())
            self.assertIn("18, 52, 171, 210", partial_label.styleSheet())
        finally:
            window.close()

    def test_canceling_custom_overlay_color_keeps_current_color(self) -> None:
        window = MainWindow()
        overlay = window._ensure_overlay()
        try:
            window.overlay_text_color_combo.setCurrentIndex(
                window.overlay_text_color_combo.findData("#FDE047")
            )
            previous_color = overlay.style_config.text_color
            previous_index = window.overlay_text_color_combo.currentIndex()

            with patch(
                "phase5.gui_app.QColorDialog.getColor", return_value=QColor()
            ):
                window.overlay_custom_color_button.click()

            self.assertEqual(window.overlay_text_color_combo.currentIndex(), previous_index)
            self.assertEqual(overlay.style_config.text_color, previous_color)
        finally:
            window.close()

    def test_long_mock_captions_expand_overlay_without_label_clipping(self) -> None:
        window = MainWindow()
        overlay = window._ensure_overlay()
        initial_height = overlay.height()
        long_text = "긴 자막이 화면 폭에 맞춰 자연스럽게 줄바꿈되는지 확인합니다. "
        try:
            window.apply_subtitle_event(self.event("speaker_0", 1, 1, "첫 번째 확정 자막", "final"))
            window.apply_subtitle_event(self.event("speaker_1", 1, 2, "두 번째 확정 자막", "final"))
            window.apply_subtitle_event(self.event("speaker_0", 2, 3, long_text))
            overlay.show()
            self.app.processEvents()
            visible = [label for label in overlay._labels if label.isVisible()]
            self.assertEqual(len(visible), overlay.MAX_ROWS)
            self.assertGreater(overlay.height(), initial_height)
            self.assertTrue(all(label.wordWrap() for label in visible))
            self.assertTrue(
                all(label.height() >= label.heightForWidth(label.width()) for label in visible)
            )
        finally:
            window.close()

    def test_overlay_height_shrinks_with_three_two_one_and_zero_rows(self) -> None:
        window = MainWindow()
        overlay = window._ensure_overlay()
        entries = tuple(
            SubtitleFeedEntry(
                speaker_id="speaker_0",
                utterance_id=index,
                sequence=index,
                status="partial",
                text=f"자막 {index}",
            )
            for index in range(1, 4)
        )
        measurements: list[tuple[int, int, int]] = []
        try:
            overlay.show()
            for row_count in (3, 2, 1, 0):
                overlay.set_entries(entries[:row_count])
                self.app.processEvents()
                metrics = overlay.layout_metrics
                measurements.append(
                    (row_count, metrics.required_height, metrics.window_height)
                )
                self.assertEqual(metrics.source_rows, row_count)
                self.assertEqual(metrics.visible_rows, row_count)
                self.assertEqual(metrics.window_height, metrics.required_height)

            heights = [actual for _rows, _required, actual in measurements]
            self.assertTrue(
                all(earlier > later for earlier, later in zip(heights, heights[1:])),
                measurements,
            )
        finally:
            window.close()

    def test_partial_overlay_uses_selected_weight_and_color(self) -> None:
        window = MainWindow()
        overlay = window._ensure_overlay()
        try:
            window.overlay_text_color_combo.setCurrentIndex(
                window.overlay_text_color_combo.findData("#A7F3D0")
            )
            window.apply_subtitle_event(self.event("speaker_0", 1, 1, "진행 중 자막"))
            label = overlay._labels[0]
            self.assertTrue(label.font().bold())
            self.assertIn("167, 243, 208", label.styleSheet())
        finally:
            window.close()

    def test_extreme_wrapped_rows_keep_only_the_newest_readable_overlay_entry(self) -> None:
        window = MainWindow()
        overlay = window._ensure_overlay()
        text = "매우 긴 자막이 여러 행으로 반복되어도 오버레이가 잘린 행을 표시하지 않아야 합니다. " * 3
        try:
            window.apply_subtitle_event(self.event("speaker_0", 1, 1, text, "final"))
            window.apply_subtitle_event(self.event("speaker_1", 1, 2, text, "final"))
            window.apply_subtitle_event(self.event("speaker_0", 2, 3, text))
            overlay.show()
            self.app.processEvents()
            visible = [label for label in overlay._labels if label.isVisible()]
            self.assertLess(len(visible), overlay.MAX_ROWS)
            self.assertEqual(overlay.entries[-1].sequence, 3)
            self.assertLessEqual(overlay.height(), overlay.layout_metrics.maximum_height)
            self.assertEqual(window.subtitle_model.rowCount(), window.subtitle_model.MAX_ENTRIES)
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

    def test_stop_before_worker_runs_skips_preflight_request(self) -> None:
        backend = Mock()
        worker = LiveCaptionWorker(LiveCaptionConfig(), backend=backend)
        finished: list[tuple[bool, str]] = []
        worker.finished.connect(lambda success, message: finished.append((success, message)))

        worker.request_stop()
        worker.run()

        backend.preflight.assert_not_called()
        self.assertEqual(finished, [(True, "시작이 취소되었습니다.")])

    def test_connection_failure_returns_idle_and_a_later_start_recovers(self) -> None:
        backend = RecoveringBackend()
        controller = LiveCaptionController(backend=backend)
        errors: list[str] = []
        controller.error.connect(errors.append)
        try:
            self.assertTrue(controller.start(LiveCaptionConfig(websocket_enabled=False)))
            self.pump_until(lambda: controller.state == "idle")
            self.assertEqual(backend.preflight_attempts, 1)
            self.assertIn("RunPod 서버에 연결할 수 없습니다", errors[-1])

            self.assertTrue(controller.start(LiveCaptionConfig(websocket_enabled=False)))
            self.pump_until(backend.started.is_set)
            self.assertEqual(backend.preflight_attempts, 2)
            self.assertTrue(controller.stop())
            self.pump_until(lambda: controller.state == "idle")
        finally:
            if controller.state != "idle":
                controller.stop()
                self.pump_until(lambda: controller.state == "idle")

    def test_connection_and_timeout_messages_are_distinct(self) -> None:
        self.assertIn(
            "연결할 수 없습니다",
            user_error_message(RemoteConnectionError("refused")),
        )
        self.assertIn(
            "응답 시간이 초과",
            user_error_message(RemoteTimeoutError("read timed out")),
        )

    def test_backend_reuses_phase4_pipeline_with_fixed_live_settings(self) -> None:
        backend = Phase4PipelineBackend()
        config = LiveCaptionConfig(
            server_url="https://example.invalid",
            output_device_id="{selected-endpoint}",
        )
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
        self.assertEqual(argv[argv.index("--output-device-id") + 1], "{selected-endpoint}")
        self.assertIn("--websocket", argv)
        self.assertNotIn("--evidence-dir", argv)
        self.assertNotIn("--latency-diagnostics", argv)
        self.assertIs(main.call_args.kwargs["runtime_hooks"], hooks)

    def test_launch_options_defaults_and_qt_arguments(self) -> None:
        config, qt_args = parse_launch_options(["-platform", "offscreen"])
        self.assertEqual(config, LiveCaptionConfig())
        self.assertEqual(qt_args, ["-platform", "offscreen"])

    def test_invalid_window_stride_rejected_before_gui_start(self) -> None:
        for window, stride in ((0, 2), (3, 0), (-1, 2), (3, -1), (3, 3), (2, 3)):
            with self.subTest(window=window, stride=stride):
                with self.assertRaises(ValueError):
                    LiveCaptionConfig(window_seconds=window, stride_seconds=stride)
                with (
                    patch("phase5.gui_app.configure_gui_runtime") as runtime,
                    patch("sys.stderr"),
                    self.assertRaises(SystemExit) as error,
                ):
                    gui_main(["--window-seconds", str(window), "--stride-seconds", str(stride)])
                self.assertEqual(error.exception.code, 2)
                runtime.assert_not_called()
        for value in (3.5, "4", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LiveCaptionConfig(window_seconds=value)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_launch_options(["--window-seconds", "3.5"])

    def test_gui_main_passes_experiment_config_and_preserves_qt_options(self) -> None:
        with (
            patch("phase5.gui_app.configure_gui_runtime"),
            patch("phase5.gui_app.shutdown_gui_runtime"),
            patch("phase5.gui_app.QApplication") as application,
            patch("phase5.gui_app.MainWindow") as window,
        ):
            application.instance.return_value = None
            application.return_value.exec.return_value = 0
            self.assertEqual(gui_main([
                "--window-seconds", "4", "--stride-seconds", "3",
                "--evidence-dir", "evidence/run B", "--latency-diagnostics",
                "-platform", "offscreen",
            ]), 0)
            config = window.call_args.kwargs["config"]
            self.assertEqual((config.window_seconds, config.stride_seconds), (4, 3))
            self.assertEqual(config.evidence_dir, Path("evidence/run B"))
            self.assertTrue(config.latency_diagnostics)
            application.assert_called_once_with([sys.argv[0], "-platform", "offscreen"])

    def test_experiment_options_survive_gui_start_restart_and_backend_adapter(self) -> None:
        config, _ = parse_launch_options([
            "--window-seconds", "4", "--stride-seconds", "3",
            "--evidence-dir", "evidence/run B", "--latency-diagnostics",
        ])
        backend = BlockingBackend()
        controller = LiveCaptionController(backend=backend)
        devices = [SimpleNamespace(device_id="{ab-device}", name="A/B", is_default=True)]
        with patch("phase5.gui_app.list_output_devices", return_value=devices):
            window = MainWindow(controller, config=config)
        try:
            window.server_url_edit.setText("https://example.invalid")
            window.output_device_combo.setCurrentIndex(1)
            for _ in range(2):
                backend.started.clear()
                window.start_captioning()
                self.pump_until(backend.started.is_set)
                received = backend.config
                self.assertEqual((received.window_seconds, received.stride_seconds), (4, 3))
                self.assertEqual(received.server_url, "https://example.invalid")
                self.assertEqual(received.output_device_id, "{ab-device}")
                self.assertEqual(received.evidence_dir, config.evidence_dir)
                self.assertTrue(received.latency_diagnostics)
                pipeline = Mock()
                pipeline.main.return_value = 0
                hooks = object()
                with patch("phase5.live_caption_controller._load_phase4_pipeline", return_value=pipeline):
                    self.assertEqual(Phase4PipelineBackend().run(received, hooks), 0)
                argv = pipeline.main.call_args.args[0]
                for option, value in (("--window-seconds", "4"), ("--stride-seconds", "3"),
                                      ("--evidence-dir", str(config.evidence_dir))):
                    self.assertEqual(argv[argv.index(option) + 1], value)
                self.assertIn("--latency-diagnostics", argv)
                self.assertIs(pipeline.main.call_args.kwargs["runtime_hooks"], hooks)
                window.stop_captioning()
                self.pump_until(lambda: controller.state == "idle")
        finally:
            if controller.state != "idle":
                window.stop_captioning()
                self.pump_until(lambda: controller.state == "idle")
            window.close()

    def test_preflight_profile_validation_rejects_known_mismatch(self) -> None:
        backend = Phase4PipelineBackend()
        with self.assertRaisesRegex(RuntimeError, "GUI=single, 서버=overlap"):
            backend._validate_server_profile(
                LiveCaptionConfig(speaker_mode=SpeakerMode.SINGLE),
                {"processing_profile": "overlap"},
            )

    def test_preflight_profile_validation_accepts_match_and_legacy_health(self) -> None:
        backend = Phase4PipelineBackend()
        config = LiveCaptionConfig(speaker_mode=SpeakerMode.SINGLE)
        backend._validate_server_profile(
            config,
            {"processing_profile": "single"},
        )
        backend._validate_server_profile(
            config,
            {"status": "ready", "gpu_ready": True, "models_loaded": True},
        )

    def test_worker_can_be_constructed(self) -> None:
        worker = LiveCaptionWorker(
            LiveCaptionConfig(websocket_enabled=False), backend=BlockingBackend()
        )
        self.assertFalse(worker.stop_event.is_set())
        worker.request_stop()
        self.assertTrue(worker.stop_event.is_set())

    def test_consoleless_runtime_redirects_output_to_log(self) -> None:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        original_handlers = LOGGER.handlers[:]
        original_logger_state = (LOGGER.level, LOGGER.propagate)
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = os.path.join(temporary_directory, "voxlux.log")
            with patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
                try:
                    configured_path = configure_gui_runtime(Path(log_path))
                    redirected_stdout, redirected_stderr = sys.stdout, sys.stderr
                    handler = next(h for h in LOGGER.handlers if h not in original_handlers)
                    log_stream = handler.stream
                    # Configuring twice must not duplicate handles or log records.
                    configure_gui_runtime(Path(log_path))
                    self.assertEqual(len(LOGGER.handlers), len(original_handlers) + 1)
                    self.assertIs(sys.stdout, redirected_stdout)
                    self.assertIs(sys.stderr, redirected_stderr)
                    print("consoleless pipeline diagnostic", flush=True)
                    sys.stderr.write("consoleless error diagnostic\\n")
                    sys.stderr.flush()
                    print("buffered stdout diagnostic", end="")
                    sys.stderr.write("buffered stderr diagnostic")
                    LOGGER.info("runtime logging diagnostic")
                finally:
                    shutdown_gui_runtime()
                self.assertIsNone(sys.stdout)
                self.assertIsNone(sys.stderr)
                self.assertTrue(redirected_stdout.closed)
                self.assertTrue(redirected_stderr.closed)
                self.assertTrue(log_stream.closed)
                self.assertIsNone(handler.stream)
            self.assertIs(sys.stdout, original_stdout)
            self.assertIs(sys.stderr, original_stderr)
            self.assertEqual(LOGGER.handlers, original_handlers)
            self.assertEqual((LOGGER.level, LOGGER.propagate), original_logger_state)
            self.assertEqual(configured_path, Path(log_path).resolve())
            contents = configured_path.read_text(encoding="utf-8")
            self.assertIn("consoleless pipeline diagnostic", contents)
            self.assertIn("consoleless error diagnostic", contents)
            self.assertIn("buffered stdout diagnostic", contents)
            self.assertIn("buffered stderr diagnostic", contents)
            self.assertEqual(contents.count("runtime logging diagnostic"), 1)
            shutdown_gui_runtime()  # Repeated shutdown is harmless.
            configured_path.unlink()  # Windows must release all three file handles.

    def test_runtime_shutdown_preserves_external_streams_and_handlers(self) -> None:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "voxlux.log"
            external_handler = logging.FileHandler(log_path, encoding="utf-8")
            LOGGER.addHandler(external_handler)
            try:
                try:
                    configure_gui_runtime(log_path)
                    self.assertIs(sys.stdout, original_stdout)
                    self.assertIs(sys.stderr, original_stderr)
                    with patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
                        configure_gui_runtime(log_path)
                        redirected_stdout, redirected_stderr = sys.stdout, sys.stderr
                finally:
                    shutdown_gui_runtime()
                self.assertIs(sys.stdout, original_stdout)
                self.assertIs(sys.stderr, original_stderr)
                self.assertTrue(redirected_stdout.closed)
                self.assertTrue(redirected_stderr.closed)
                self.assertIn(external_handler, LOGGER.handlers)
                self.assertFalse(external_handler.stream.closed)
            finally:
                LOGGER.removeHandler(external_handler)
                external_handler.close()

    def test_gui_main_releases_runtime_on_exit_and_startup_failure(self) -> None:
        for startup_failure in (False, True):
            with (
                self.subTest(startup_failure=startup_failure),
                tempfile.TemporaryDirectory() as directory,
            ):
                log_path = Path(directory) / "voxlux.log"
                original_handlers = LOGGER.handlers[:]
                with (
                    patch.object(sys, "stdout", None),
                    patch.object(sys, "stderr", None),
                    patch(
                        "phase5.gui_app.configure_gui_runtime",
                        side_effect=lambda: configure_gui_runtime(log_path),
                    ),
                    patch("phase5.gui_app.QApplication") as application,
                    patch("phase5.gui_app.MainWindow") as window,
                ):
                    application.instance.return_value.exec.side_effect = (
                        lambda: LOGGER.info("event loop diagnostic") or 0
                    )
                    if startup_failure:
                        window.side_effect = RuntimeError("startup diagnostic")
                    try:
                        if startup_failure:
                            with self.assertRaisesRegex(RuntimeError, "startup diagnostic"):
                                gui_main()
                        else:
                            self.assertEqual(gui_main(), 0)
                        self.assertIsNone(sys.stdout)
                        self.assertIsNone(sys.stderr)
                        self.assertEqual(LOGGER.handlers, original_handlers)
                        contents = log_path.read_text(encoding="utf-8")
                        self.assertIn(
                            "startup diagnostic" if startup_failure else "event loop diagnostic",
                            contents,
                        )
                        if startup_failure:
                            self.assertIn("VOXLUX GUI startup failed", contents)
                        log_path.unlink()
                    finally:
                        shutdown_gui_runtime()


if __name__ == "__main__":
    unittest.main()
