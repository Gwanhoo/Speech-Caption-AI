from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from websockets.asyncio.server import Server, ServerConnection, serve


STOP = object()


@dataclass(frozen=True)
class PublishResult:
    accepted: bool
    dropped_oldest: bool
    queue_depth: int


class SubtitleWebSocketServer:
    """Thread-isolated WebSocket broadcaster for non-blocking AI event publication."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, queue_maxsize: int = 32) -> None:
        if queue_maxsize < 1:
            raise ValueError("queue_maxsize must be positive")
        self.host = host
        self.port = port
        self.event_queue: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=queue_maxsize)
        self._queue_maxsize = queue_maxsize
        self._clients: set[ServerConnection] = set()
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None
        self._server: Server | None = None
        self._running = False
        self._stats: dict[str, Any] = {
            "published": 0,
            "dropped_oldest": 0,
            "broadcast": 0,
            "send_failures": 0,
            "client_connects": 0,
            "client_disconnects": 0,
            "max_queue_depth": 0,
            "enqueue_to_send_ms": [],
            "browser_receive_ms": [],
            "last_sequence": None,
        }

    def start(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            raise RuntimeError("WebSocket server is already started")
        self._running = True
        self._thread = threading.Thread(target=self._run, name="subtitle_websocket", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("Timed out starting WebSocket server")
        if self._startup_error:
            raise RuntimeError("WebSocket server failed to start") from self._startup_error

    def publish(self, event: dict[str, Any]) -> PublishResult:
        """Accept an event without waiting for socket I/O; retain the latest subtitle on pressure."""
        if not self._running:
            return PublishResult(False, False, self.event_queue.qsize())
        event = dict(event)
        event["enqueued_at_ms"] = int(time.time() * 1000)
        dropped_oldest = False
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            try:
                self.event_queue.get_nowait()
                self.event_queue.task_done()
                dropped_oldest = True
            except queue.Empty:
                return PublishResult(False, False, self.event_queue.qsize())
            try:
                self.event_queue.put_nowait(event)
            except queue.Full:
                return PublishResult(False, dropped_oldest, self.event_queue.qsize())
        with self._lock:
            self._stats["published"] += 1
            self._stats["last_sequence"] = event.get("sequence")
            if dropped_oldest:
                self._stats["dropped_oldest"] += 1
            self._stats["max_queue_depth"] = max(
                self._stats["max_queue_depth"], self.event_queue.qsize()
            )
        return PublishResult(True, dropped_oldest, self.event_queue.qsize())

    def flush(self, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while self.event_queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        return self.event_queue.unfinished_tasks == 0

    def stop(self, timeout: float = 5.0) -> None:
        if not self._running:
            return
        self._running = False
        while True:
            try:
                self.event_queue.put_nowait(STOP)
                break
            except queue.Full:
                try:
                    self.event_queue.get_nowait()
                    self.event_queue.task_done()
                except queue.Empty:
                    break
        if self._thread:
            self._thread.join(timeout)
        self._stopped.set()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            snapshot = dict(self._stats)
        snapshot["queue_depth"] = self.event_queue.qsize()
        snapshot["queue_maxsize"] = self._queue_maxsize
        snapshot["connected_clients"] = len(self._clients)
        snapshot["enqueue_to_send_ms"] = _distribution(snapshot["enqueue_to_send_ms"])
        snapshot["browser_receive_ms"] = _distribution(snapshot["browser_receive_ms"])
        return snapshot

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
        finally:
            self._stopped.set()

    async def _serve(self) -> None:
        async with serve(self._handle_client, self.host, self.port) as server:
            self._server = server
            self._ready.set()
            while True:
                event = await asyncio.to_thread(self.event_queue.get)
                try:
                    if event is STOP:
                        break
                    assert isinstance(event, dict)
                    await self._broadcast(event)
                finally:
                    self.event_queue.task_done()

    async def _handle_client(self, websocket: ServerConnection) -> None:
        self._clients.add(websocket)
        with self._lock:
            self._stats["client_connects"] += 1
        try:
            async for message in websocket:
                if isinstance(message, str):
                    self._record_receipt(message)
        finally:
            self._clients.discard(websocket)
            with self._lock:
                self._stats["client_disconnects"] += 1

    def _record_receipt(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return
        if payload.get("type") != "receipt":
            return
        sent_at_ms = payload.get("sent_at_ms")
        received_at_ms = payload.get("received_at_ms")
        if not isinstance(sent_at_ms, (int, float)) or not isinstance(received_at_ms, (int, float)):
            return
        latency = max(0.0, received_at_ms - sent_at_ms)
        with self._lock:
            self._stats["browser_receive_ms"].append(latency)

    async def _broadcast(self, event: dict[str, Any]) -> None:
        event["sent_at_ms"] = int(time.time() * 1000)
        payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        enqueue_at_ms = event.get("enqueued_at_ms")
        if isinstance(enqueue_at_ms, (int, float)):
            with self._lock:
                self._stats["enqueue_to_send_ms"].append(max(0.0, event["sent_at_ms"] - enqueue_at_ms))
        clients = tuple(self._clients)
        if not clients:
            return
        sends = [asyncio.wait_for(client.send(payload), timeout=0.25) for client in clients]
        results = await asyncio.gather(*sends, return_exceptions=True)
        failed = [client for client, result in zip(clients, results) if isinstance(result, BaseException)]
        for client in failed:
            self._clients.discard(client)
            try:
                await client.close()
            except BaseException:
                pass
        with self._lock:
            self._stats["broadcast"] += 1
            self._stats["send_failures"] += len(failed)


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * 0.95))
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p95": ordered[index],
        "max": ordered[-1],
    }
