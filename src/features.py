from __future__ import annotations

import torch
import torchaudio
import torchaudio.transforms as T
from torch import Tensor

from .utils import FeatureConfig


def extract_features(waveform: Tensor, sample_rate: int, cfg: FeatureConfig) -> Tensor:
    """Return feature tensor of shape (T, feature_dim)."""
    if cfg.feature_type == "mfcc":
        return _extract_mfcc(waveform, sample_rate, cfg)
    elif cfg.feature_type == "mel":
        return _extract_mel(waveform, sample_rate, cfg)
    else:
        raise ValueError(f"Unknown feature_type: {cfg.feature_type}")


def _extract_mfcc(waveform: Tensor, sample_rate: int, cfg: FeatureConfig) -> Tensor:
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=cfg.n_mfcc,
        melkwargs={
            "n_fft": cfg.n_fft,
            "hop_length": cfg.hop_length,
            "n_mels": cfg.n_mels,
        },
    )
    mfcc = mfcc_transform(waveform).squeeze(0)  # (n_mfcc, T)
    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)
    features = torch.cat([mfcc, delta, delta2], dim=0)  # (n_mfcc*3, T)
    return features.T                                    # (T, n_mfcc*3)


def _extract_mel(waveform: Tensor, sample_rate: int, cfg: FeatureConfig) -> Tensor:
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
    )
    mel = mel_transform(waveform).squeeze(0)    # (n_mels, T)
    log_mel = torch.log(mel + 1e-9)
    return log_mel.T                             # (T, n_mels)
