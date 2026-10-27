"""Bounded offline check of installed models; never downloads or opens a device."""
import json
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'phase3'), str(ROOT / 'phase4')]
import gpu_processing_server as server
from stt_context import transcribe_base


def main():
    output = dict(environment={name: version(name) for name in
                  ('torch', 'silero-vad', 'faster-whisper')}, cases=[])
    output['cuda_available'] = torch.cuda.is_available()
    snapshots = ROOT / 'checkpoints/faster-whisper/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots'
    local = next((p for p in snapshots.glob('*') if (p / 'model.bin').is_file()), None)
    vad = load_silero_vad()
    silence = np.zeros(48000, np.float32)
    noise = np.random.default_rng(20261027).normal(0, 1e-4, 48000).astype(np.float32)
    speech, rate = sf.read(ROOT / 'phase1/reference/speaker_a.wav', dtype='float32')
    assert rate == 16000 and speech.ndim == 1 and len(speech) >= 80000
    cases = [('digital_silence', silence), ('low_amplitude_noise', noise),
             ('reference_0_3s', speech[:48000]), ('silence_after_speech', silence),
             ('reference_2_5s', speech[32000:80000]), ('silence_again', silence)]
    for name, audio in cases:
        result = server.detect_speech_activity(vad, audio, 200)
        output['cases'].append(dict(name=name, vad=result))
        if 'silence' in name or 'noise' in name:
            assert not result['speech_detected'], (name, result)
    output['real_vad_status'] = 'PASS'
    if local is None or not torch.cuda.is_available():
        output['gpu_status'] = 'SKIP'
        output['gpu_reason'] = 'Local model or CUDA unavailable; no download attempted'
    else:
        whisper = WhisperModel(str(local), device='cuda', compute_type='float16', local_files_only=True)
        models = server.LoadedModels(None, None, whisper, vad)
        with patch.object(server.PipelineService, '_load_models', return_value=models):
            service = server.PipelineService(warm_up=False)
        try:
            for record, (name, audio) in zip(output['cases'], cases):
                result = service.process(name, audio, 3.)
                record['server_slot'] = {k: v for k, v in result['speakers'][0].items() if k != 'waveform'}
                if 'silence' in name or 'noise' in name:
                    assert not record['server_slot']['stt_diagnostics']['executed']
                    assert not record['server_slot']['raw_transcript']
                else:
                    assert record['server_slot']['stt_diagnostics']['executed']
                    assert record['server_slot']['raw_transcript']
            # Diagnostic-only bypass: distinguish model output from gated server output.
            output['ungated_silence_stt'] = asdict(transcribe_base(whisper, silence))
            output['ungated_noise_stt'] = asdict(transcribe_base(whisper, noise))
            output['gpu_status'] = 'PASS'
        finally:
            service.close()
    (ROOT / 'docs/p0_2_evidence/real_models.json').write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in output.items() if k != 'cases'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
