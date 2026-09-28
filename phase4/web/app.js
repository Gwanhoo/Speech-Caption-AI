const MAX_EVENTS_PER_SPEAKER = 4;
const wsUrl = new URLSearchParams(window.location.search).get("ws") || "ws://127.0.0.1:8765";
const statusElement = document.getElementById("connection-status");
const lastSequenceElement = document.getElementById("last-sequence");
const lastWindowElement = document.getElementById("last-window");
const lastLatencyElement = document.getElementById("last-latency");

let socket;
let retryDelayMs = 500;
let lastSequence = 0;

function setStatus(text, state) {
  statusElement.textContent = text;
  statusElement.className = `status status-${state}`;
}

function addSubtitle(event) {
  if (event.type !== "subtitle" || !event.text || event.sequence <= lastSequence) return;
  const list = document.getElementById(event.speaker);
  if (!list) return;
  lastSequence = event.sequence;
  const utteranceKey = event.utterance_id == null
    ? null
    : `${event.speaker}:${event.utterance_id}`;
  let item = utteranceKey
    ? [...list.children].find((candidate) => candidate.dataset.utteranceKey === utteranceKey)
    : null;
  if (!item) {
    item = document.createElement("li");
    if (utteranceKey) item.dataset.utteranceKey = utteranceKey;
    list.append(item);
  }
  item.dataset.status = event.status || "final";
  item.replaceChildren();
  const time = document.createElement("time");
  time.textContent = `${event.start_ms}ms - ${event.end_ms}ms`;
  item.append(time, document.createTextNode(event.text));
  while (list.children.length > MAX_EVENTS_PER_SPEAKER) list.removeChild(list.firstChild);
  const receiveLatency = Math.max(0, Date.now() - event.sent_at_ms);
  lastSequenceElement.textContent = `Last sequence: ${event.sequence}`;
  lastWindowElement.textContent = `Window: ${event.window_index}`;
  lastLatencyElement.textContent = `Browser receive: ${receiveLatency}ms`;
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({
      type: "receipt",
      sequence: event.sequence,
      sent_at_ms: event.sent_at_ms,
      received_at_ms: Date.now(),
    }));
  }
}

function connect() {
  setStatus(lastSequence ? "재연결 중..." : "연결 중...", lastSequence ? "reconnecting" : "connecting");
  socket = new WebSocket(wsUrl);
  socket.addEventListener("open", () => {
    retryDelayMs = 500;
    setStatus("연결됨", "connected");
  });
  socket.addEventListener("message", ({ data }) => {
    try { addSubtitle(JSON.parse(data)); } catch { /* Ignore malformed messages. */ }
  });
  socket.addEventListener("close", () => {
    setStatus("연결 끊김", "disconnected");
    window.setTimeout(connect, retryDelayMs);
    retryDelayMs = Math.min(retryDelayMs * 2, 5000);
  });
  socket.addEventListener("error", () => socket.close());
}

connect();
