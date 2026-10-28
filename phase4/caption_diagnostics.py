"""Opt-in append-only caption tracing. Never pass URLs, headers or exceptions here."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

LOGGER = logging.getLogger("voxlux.caption_trace")
LOGGER.propagate = False
LOGGER.setLevel(logging.WARNING)


class CaptionTraceHandler(logging.Handler):
    def __init__(self, path: Path):
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("x", encoding="utf-8")
        self.failed = False

    def emit(self, record):
        payload = getattr(record, "caption_trace", None)
        if payload is None or self.failed:
            return
        try:
            self.stream.write(json.dumps({
                "schema_version": 1, "wall_time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(), **payload,
            }, ensure_ascii=False, allow_nan=False) + "\n")
            self.stream.flush()
        except (OSError, TypeError, ValueError):
            self.failed = True
            logging.getLogger("voxlux").warning(
                "Caption trace disabled after a write/serialization failure")

    def close(self):
        try:
            self.stream.close()
        finally:
            super().close()


def trace_event(stage: str, **fields):
    if LOGGER.isEnabledFor(logging.INFO):
        LOGGER.info("caption", extra={"caption_trace": {"stage": stage, **fields}})


def start_trace(path: Path):
    """Return an owned handler, or None when the GUI already owns this trace."""
    if LOGGER.handlers:
        return None
    handler = CaptionTraceHandler(path)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    return handler


def stop_trace(handler):
    if handler is not None:
        LOGGER.removeHandler(handler)
        handler.close()
        if not LOGGER.handlers:
            LOGGER.setLevel(logging.WARNING)
