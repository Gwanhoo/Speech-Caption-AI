"""Windows output-device selection helpers for WASAPI loopback capture.

``soundcard`` exposes stable WASAPI endpoint identifiers on speaker objects.
Names are intentionally only presentation data: Windows may expose more than
one output endpoint with the same friendly name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_OUTPUT_DEVICE_ID: None = None


@dataclass(frozen=True)
class AudioOutputDevice:
    """A selectable, currently available Windows output endpoint."""

    device_id: str
    name: str
    is_default: bool
    channels: int | None


def _device_id(speaker: Any) -> str | None:
    identifier = getattr(speaker, "id", None)
    if identifier is None:
        return None
    value = str(identifier).strip()
    return value or None


def list_output_devices(soundcard: Any) -> list[AudioOutputDevice]:
    """Return available speakers with stable IDs, default endpoint first."""
    default = soundcard.default_speaker()
    default_id = _device_id(default) if default is not None else None
    devices: list[AudioOutputDevice] = []
    seen_ids: set[str] = set()
    for speaker in soundcard.all_speakers():
        identifier = _device_id(speaker)
        if identifier is None or identifier in seen_ids:
            continue
        seen_ids.add(identifier)
        name = str(getattr(speaker, "name", "")).strip() or identifier
        channels = getattr(speaker, "channels", None)
        devices.append(
            AudioOutputDevice(
                device_id=identifier,
                name=name,
                is_default=identifier == default_id,
                channels=channels if isinstance(channels, int) else None,
            )
        )
    return sorted(devices, key=lambda device: (not device.is_default, device.name.casefold(), device.device_id))


def resolve_output_speaker(soundcard: Any, output_device_id: str | None) -> Any:
    """Resolve the default endpoint or an explicitly selected WASAPI ID."""
    if output_device_id is None:
        speaker = soundcard.default_speaker()
        if speaker is None:
            raise RuntimeError("기본 Windows 출력 오디오 장치를 찾을 수 없습니다.")
        if _device_id(speaker) is None:
            raise RuntimeError("기본 Windows 출력 장치에 WASAPI 식별자가 없습니다.")
        return speaker

    requested_id = str(output_device_id).strip()
    if not requested_id:
        raise RuntimeError("선택한 Windows 출력 장치 식별자가 비어 있습니다.")
    for speaker in soundcard.all_speakers():
        if _device_id(speaker) == requested_id:
            return speaker
    raise RuntimeError("선택한 Windows 출력 장치를 찾을 수 없습니다. 장치 연결을 확인한 뒤 다시 시작하세요.")


def get_loopback_microphone(soundcard: Any, speaker: Any) -> Any:
    """Resolve loopback by endpoint ID, never by its non-unique friendly name."""
    identifier = _device_id(speaker)
    if identifier is None:
        raise RuntimeError("선택한 Windows 출력 장치에 WASAPI 식별자가 없습니다.")
    loopback = soundcard.get_microphone(identifier, include_loopback=True)
    if loopback is None:
        raise RuntimeError("선택한 출력 장치의 WASAPI loopback을 찾을 수 없습니다.")
    return loopback
