from __future__ import annotations

import unittest

try:
    from audio_devices import get_loopback_microphone, list_output_devices, resolve_output_speaker
except ImportError:  # Repository-root unittest discovery.
    from phase3.audio_devices import get_loopback_microphone, list_output_devices, resolve_output_speaker


class Speaker:
    def __init__(self, identifier: str, name: str, channels: int = 2) -> None:
        self.id = identifier
        self.name = name
        self.channels = channels


class SoundcardStub:
    def __init__(self, speakers: list[Speaker], default: Speaker | None) -> None:
        self.speakers = speakers
        self.default = default
        self.loopback_requests: list[tuple[str, bool]] = []

    def all_speakers(self) -> list[Speaker]:
        return list(self.speakers)

    def default_speaker(self) -> Speaker | None:
        return self.default

    def get_microphone(self, identifier: str, include_loopback: bool = False) -> object:
        self.loopback_requests.append((identifier, include_loopback))
        return object()


class AudioOutputDeviceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.default = Speaker("{default-id}", "같은 이름")
        self.second = Speaker("{second-id}", "같은 이름", 8)
        self.soundcard = SoundcardStub([self.second, self.default], self.default)

    def test_lists_available_devices_with_unique_ids_when_names_duplicate(self) -> None:
        devices = list_output_devices(self.soundcard)
        self.assertEqual([device.device_id for device in devices], ["{default-id}", "{second-id}"])
        self.assertEqual([device.name for device in devices], ["같은 이름", "같은 이름"])
        self.assertTrue(devices[0].is_default)
        self.assertEqual(devices[1].channels, 8)

    def test_default_selection_resolves_current_default(self) -> None:
        self.assertIs(resolve_output_speaker(self.soundcard, None), self.default)

    def test_explicit_selection_uses_id_not_duplicate_name(self) -> None:
        selected = resolve_output_speaker(self.soundcard, "{second-id}")
        self.assertIs(selected, self.second)
        get_loopback_microphone(self.soundcard, selected)
        self.assertEqual(self.soundcard.loopback_requests, [("{second-id}", True)])

    def test_missing_selected_device_reports_actionable_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "선택한 Windows 출력 장치를 찾을 수 없습니다"):
            resolve_output_speaker(self.soundcard, "{removed-id}")

    def test_missing_default_device_reports_actionable_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "기본 Windows 출력 오디오 장치를 찾을 수 없습니다"):
            resolve_output_speaker(SoundcardStub([], None), None)


if __name__ == "__main__":
    unittest.main()
