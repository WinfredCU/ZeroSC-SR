from __future__ import annotations
from typing import Tuple
import torch
import torchaudio
import soundfile as sf


def rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x**2) + 1e-12)


def load_audio(path: str, target_sr: int) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    wav = torch.clamp(wav, -1.0, 1.0)
    return wav, sr


def normalize_rms(wav: torch.Tensor, target_rms: float) -> torch.Tensor:
    cur = rms(wav)
    if cur.item() > 0:
        wav = wav * (target_rms / cur)
    wav = torch.clamp(wav, -1.0, 1.0)
    return wav


def save_wav(path: str, wav: torch.Tensor, sr: int) -> None:
    wav_np = wav.detach().cpu().numpy()
    if wav_np.ndim == 2:
        wav_np = wav_np.transpose(1, 0)
    sf.write(path, wav_np, sr, subtype="PCM_16")
