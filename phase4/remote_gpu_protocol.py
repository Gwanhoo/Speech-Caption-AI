from __future__ import annotations

import json
import struct
from typing import Any


BINARY_RESPONSE_MEDIA_TYPE = (
    "application/vnd.speech-caption.processing-result.v1+octet-stream"
)
BINARY_RESPONSE_MAGIC = b"SCP1"
_HEADER = struct.Struct("!4sI")
_MAX_METADATA_BYTES = 1024 * 1024


def encode_binary_result(payload: dict[str, Any]) -> bytes:
    metadata = dict(payload)
    metadata_speakers = []
    waveform_parts: list[bytes] = []
    offset = 0
    for slot in payload["speakers"]:
        waveform = slot["waveform"]
        data = waveform["data"]
        if waveform["encoding"] != "binary-f32le" or not isinstance(data, bytes):
            raise ValueError("Binary response requires f32le waveform bytes")
        metadata_slot = dict(slot)
        metadata_slot["waveform"] = {
            "encoding": "binary-f32le",
            "offset": offset,
            "byte_length": len(data),
        }
        metadata_speakers.append(metadata_slot)
        waveform_parts.append(data)
        offset += len(data)
    metadata["speakers"] = metadata_speakers
    metadata_bytes = json.dumps(
        metadata, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    if len(metadata_bytes) > _MAX_METADATA_BYTES:
        raise ValueError("Binary response metadata is too large")
    return _HEADER.pack(BINARY_RESPONSE_MAGIC, len(metadata_bytes)) + metadata_bytes + b"".join(
        waveform_parts
    )


def decode_binary_result(body: bytes) -> dict[str, Any]:
    if len(body) < _HEADER.size:
        raise ValueError("Binary response is truncated")
    magic, metadata_length = _HEADER.unpack_from(body)
    if magic != BINARY_RESPONSE_MAGIC:
        raise ValueError("Invalid binary response magic")
    if metadata_length > _MAX_METADATA_BYTES:
        raise ValueError("Binary response metadata is too large")
    metadata_end = _HEADER.size + metadata_length
    if metadata_end > len(body):
        raise ValueError("Binary response metadata is truncated")
    payload = json.loads(body[_HEADER.size:metadata_end].decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Binary response metadata must be an object")
    waveform_body = body[metadata_end:]
    consumed = 0
    for slot in sorted(payload["speakers"], key=lambda item: item["raw_slot"]):
        waveform = slot["waveform"]
        offset = waveform["offset"]
        byte_length = waveform["byte_length"]
        if (
            type(offset) is not int
            or type(byte_length) is not int
            or offset != consumed
            or byte_length < 0
            or offset + byte_length > len(waveform_body)
        ):
            raise ValueError("Invalid binary waveform range")
        slot["waveform"] = {
            "encoding": "binary-f32le",
            "data": waveform_body[offset:offset + byte_length],
        }
        consumed += byte_length
    if consumed != len(waveform_body):
        raise ValueError("Unexpected trailing binary waveform data")
    return payload
