from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    frame_length_ms: int = 2000
    hop_length_ms: int = 500


@dataclass
class FeatureConfig:
    feature_type: str = "mfcc"
    n_mfcc: int = 40
    n_mels: int = 128
    n_fft: int = 512
    hop_length: int = 160


@dataclass
class ModelConfig:
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.4
    bidirectional: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 32
    max_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 10
    seed: int = 42


@dataclass
class SplitConfig:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        cfg = cls()
        if "audio" in data:
            cfg.audio = AudioConfig(**data["audio"])
        if "features" in data:
            cfg.features = FeatureConfig(**data["features"])
        if "model" in data:
            cfg.model = ModelConfig(**data["model"])
        if "training" in data:
            cfg.training = TrainingConfig(**data["training"])
        if "split" in data:
            cfg.split = SplitConfig(**data["split"])
        return cfg


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger
