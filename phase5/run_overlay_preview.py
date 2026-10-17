"""Development-only visual preview for the production subtitle overlay.

This module deliberately does not construct ``MainWindow`` or a live-caption
controller.  It is safe to run while the remote GPU server is unavailable:

    .\\.venv\\Scripts\\python.exe -m phase5.run_overlay_preview
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

try:
    from .subtitle_overlay import OverlayStyle, SubtitleOverlay
    from .subtitle_presentation import SubtitlePresentationState
except ImportError:  # Direct execution: python phase5/run_overlay_preview.py
    from subtitle_overlay import OverlayStyle, SubtitleOverlay  # type: ignore[no-redef]
    from subtitle_presentation import SubtitlePresentationState  # type: ignore[no-redef]


PREVIEW_COLORS = {
    "white": "#FFFFFF",
    "yellow": "#FDE047",
    "mint": "#A7F3D0",
}


def _event(
    speaker: str,
    utterance_id: int,
    sequence: int,
    text: str,
    status: str,
) -> dict[str, object]:
    return {
        "speaker": speaker,
        "utterance_id": utterance_id,
        "sequence": sequence,
        "status": status,
        "text": text,
    }


def populate_preview(state: SubtitlePresentationState, scenario: str) -> None:
    """Feed a representative snapshot through the real presentation state."""
    state.clear()
    if scenario == "standard":
        events = (
            _event(
                "speaker_0",
                1,
                1,
                "첫 번째 FINAL 자막입니다. 창의 폭에 맞춰 자연스럽게 여러 줄로 표시되는지 확인합니다.",
                "final",
            ),
            _event(
                "speaker_1",
                1,
                2,
                "두 번째 FINAL 자막입니다. 앞 문장을 가리지 않고 세 줄 안에서 읽기 쉬운 순서로 유지되어야 합니다.",
                "final",
            ),
            _event(
                "speaker_0",
                2,
                3,
                "현재 PARTIAL 자막입니다. 같은 발화가 갱신되어도 중복 행이 생기지 않는지 확인합니다.",
                "partial",
            ),
        )
    elif scenario == "overflow":
        long_prefix = "이 자막은 화면 높이를 넘는 여러 줄 상황을 재현합니다. "
        events = (
            _event("speaker_0", 1, 1, long_prefix * 8, "final"),
            _event("speaker_1", 1, 2, long_prefix * 8, "final"),
            _event(
                "speaker_0",
                2,
                3,
                ("가장 최신 PARTIAL 자막은 가능한 범위에서 읽을 수 있도록 남아야 합니다. ") * 8,
                "partial",
            ),
        )
    else:
        raise ValueError(f"unsupported preview scenario: {scenario}")

    for event in events:
        if not state.apply(event):
            raise RuntimeError(f"preview event was rejected: {event!r}")


class OverlayPreviewSession:
    """Own a production overlay and refresh it when FINAL captions expire."""

    def __init__(self, scenario: str, style: OverlayStyle) -> None:
        self.state = SubtitlePresentationState()
        self.overlay = SubtitleOverlay()
        self.overlay.apply_style(style)
        populate_preview(self.state, scenario)
        self.overlay.set_entries(self.state.entries)

        self._expiry_timer = QTimer(self.overlay)
        self._expiry_timer.setInterval(100)
        self._expiry_timer.timeout.connect(self._expire_finals)
        self._expiry_timer.start()

    def show(self) -> None:
        self.overlay.show()
        self.overlay.raise_()
        QTimer.singleShot(0, self._report_geometry)

    def close(self) -> None:
        self._expiry_timer.stop()
        self.overlay.close()

    def _expire_finals(self) -> None:
        if self.state.expire():
            self.overlay.set_entries(self.state.entries)
            self._report_geometry()

    def _report_geometry(self) -> None:
        metrics = self.overlay.layout_metrics
        print(
            "Overlay geometry: "
            f"source_rows={metrics.source_rows}, visible_rows={metrics.visible_rows}, "
            f"required={metrics.required_height}px, maximum={metrics.maximum_height}px, "
            f"actual={metrics.window_height}px"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the VOXLUX production subtitle overlay without audio or a server."
    )
    parser.add_argument("--scenario", choices=("standard", "overflow"), default="standard")
    parser.add_argument("--color", choices=tuple(PREVIEW_COLORS), default="yellow")
    parser.add_argument("--font-size", type=int, default=34)
    parser.add_argument("--opacity", type=int, default=72)
    parser.add_argument("--no-bold", action="store_true", help="Preview regular instead of bold text")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv)
    session = OverlayPreviewSession(
        args.scenario,
        OverlayStyle(
            font_size=args.font_size,
            background_opacity=args.opacity,
            bold=not args.no_bold,
            text_color=PREVIEW_COLORS[args.color],
        ),
    )
    session.show()
    print(
        "VOXLUX overlay preview is running (no RunPod, audio capture, or STT). "
        "Close the overlay with Alt+F4. FINAL rows expire after four seconds."
    )
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
