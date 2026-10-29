import math
from typing import Protocol

import numpy as np
import torch

F0_MIN = 50
F0_MAX = 1100
F0_MEL_MIN = 1127 * math.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * math.log(1 + F0_MAX / 700)


class PitchExtractor(Protocol):
    # True: extract() takes and returns torch tensors on the pipeline device (no host round trip).
    # False (legacy CPU extractors): pitchf is a numpy array and numpy arrays are returned.
    gpu_native: bool = False

    def extract(self, audio, pitchf, f0_up_key, sr, window, silence_front=0):
        ...

    def getPitchExtractorInfo(self):
        return {
            "pitchExtractorType": self.pitchExtractorType,
        }


def place_tail(pitchf: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    """Write freshly estimated f0 over the end of the f0 history (same alignment as upstream)."""
    n = pitchf.shape[0]
    if f0.shape[0] >= n:
        pitchf.copy_(f0[:n])
    else:
        pitchf[-f0.shape[0]:] = f0
    return pitchf


def pitch_scale(f0_up_key):
    """Semitone shift as a multiplier; stays a tensor (no host sync) when the key is a tensor."""
    if torch.is_tensor(f0_up_key):
        return torch.pow(2.0, f0_up_key.float() / 12)
    return 2 ** (float(f0_up_key) / 12)


def coarse_pitch(f0: torch.Tensor) -> torch.Tensor:
    f0_mel = 1127 * torch.log1p(f0 / 700)
    f0_mel = torch.where(f0_mel > 0, (f0_mel - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1, f0_mel)
    return torch.round(f0_mel.clamp(1, 255)).long()


def analysis_window(audio_len: int, silence_front: float, sr: int, window: int, context: int, min_len: int) -> int:
    """Start sample for f0 analysis: skip the already-analysed history but keep some left context."""
    start = int(silence_front * sr / window) * window
    start = max(0, start - context)
    return max(0, min(start, audio_len - min_len))


def legacy_extract(extractor, audio, pitchf: torch.Tensor, f0_up_key, sr, window, silence_front):
    """Adapter so numpy/CPU extractors (dio, harvest, crepe, onnx) work with the GPU pipeline."""
    coarse, f0 = extractor.extract(audio, pitchf.detach().cpu().numpy().astype(np.float64), f0_up_key, sr, window, silence_front=silence_front)
    dev = pitchf.device
    return torch.as_tensor(np.asarray(coarse), device=dev).long(), torch.as_tensor(np.asarray(f0), device=dev, dtype=torch.float32)
