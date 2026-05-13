from __future__ import annotations

import torch
import pytest

from src.features import extract_features
from src.utils import FeatureConfig


SAMPLE_RATE = 16000


@pytest.fixture
def waveform() -> torch.Tensor:
    return torch.randn(1, SAMPLE_RATE * 2)  # 2 seconds


def test_mfcc_shape(waveform: torch.Tensor) -> None:
    cfg = FeatureConfig(feature_type="mfcc", n_mfcc=40, n_fft=512, hop_length=160)
    features = extract_features(waveform, SAMPLE_RATE, cfg)
    assert features.ndim == 2
    assert features.shape[1] == 120  # 40 * 3 (mfcc + delta + delta2)


def test_mel_shape(waveform: torch.Tensor) -> None:
    cfg = FeatureConfig(feature_type="mel", n_mels=128, n_fft=512, hop_length=160)
    features = extract_features(waveform, SAMPLE_RATE, cfg)
    assert features.ndim == 2
    assert features.shape[1] == 128


def test_features_finite(waveform: torch.Tensor) -> None:
    cfg = FeatureConfig(feature_type="mfcc", n_mfcc=40, n_fft=512, hop_length=160)
    features = extract_features(waveform, SAMPLE_RATE, cfg)
    assert torch.isfinite(features).all()


def test_silent_audio() -> None:
    silent = torch.zeros(1, SAMPLE_RATE)
    cfg = FeatureConfig(feature_type="mfcc", n_mfcc=40, n_fft=512, hop_length=160)
    features = extract_features(silent, SAMPLE_RATE, cfg)
    assert features.ndim == 2
    assert torch.isfinite(features).all()


def test_unknown_feature_type_raises(waveform: torch.Tensor) -> None:
    cfg = FeatureConfig(feature_type="unknown")
    with pytest.raises(ValueError, match="Unknown feature_type"):
        extract_features(waveform, SAMPLE_RATE, cfg)
