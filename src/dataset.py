from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .features import extract_features
from .utils import AudioConfig, Config, FeatureConfig

logger = logging.getLogger(__name__)


class AudioFrameDataset(Dataset):
    """Yields (feature_tensor, label) pairs for individual audio frames."""

    def __init__(
        self,
        recordings_dir: Path,
        file_labels: list[tuple[str, int]],
        audio_cfg: AudioConfig,
        feature_cfg: FeatureConfig,
    ) -> None:
        self.recordings_dir = recordings_dir
        self._frames: list[tuple[Tensor, int]] = []

        frame_samples = int(audio_cfg.sample_rate * audio_cfg.frame_length_ms / 1000)
        hop_samples = int(audio_cfg.sample_rate * audio_cfg.hop_length_ms / 1000)

        for filename, label in file_labels:
            path = recordings_dir / filename
            try:
                waveform, sr = torchaudio.load(str(path))
            except Exception as exc:
                logger.warning("Cannot load %s: %s", path, exc)
                continue

            if sr != audio_cfg.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, audio_cfg.sample_rate)
            waveform = waveform.mean(dim=0, keepdim=True)  # mono

            total = waveform.shape[1]
            start = 0
            while start < total:
                end = start + frame_samples
                frame = waveform[:, start:end]
                if frame.shape[1] < frame_samples:
                    pad = frame_samples - frame.shape[1]
                    frame = torch.nn.functional.pad(frame, (0, pad))
                features = extract_features(frame, audio_cfg.sample_rate, feature_cfg)
                self._frames.append((features, label))
                start += hop_samples

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        return self._frames[idx]


def _collate_fn(batch: list[tuple[Tensor, int]]) -> tuple[Tensor, Tensor]:
    features, labels = zip(*batch)
    max_len = max(f.shape[0] for f in features)
    padded = torch.zeros(len(features), max_len, features[0].shape[1])
    for i, f in enumerate(features):
        padded[i, : f.shape[0]] = f
    return padded, torch.tensor(labels, dtype=torch.long)


def make_splits(
    labels_csv: Path,
    cfg: Config,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    df = pd.read_csv(labels_csv)
    rng = np.random.default_rng(cfg.training.seed)
    indices = rng.permutation(len(df))
    n_train = int(len(df) * cfg.split.train)
    n_val = int(len(df) * cfg.split.val)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def rows(idx: np.ndarray) -> list[tuple[str, int]]:
        return [(df.iloc[i]["filename"], int(df.iloc[i]["label"])) for i in idx]

    return rows(train_idx), rows(val_idx), rows(test_idx)


def make_dataloaders(
    recordings_dir: Path,
    labels_csv: Path,
    cfg: Config,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_files, val_files, test_files = make_splits(labels_csv, cfg)

    train_ds = AudioFrameDataset(recordings_dir, train_files, cfg.audio, cfg.features)
    val_ds = AudioFrameDataset(recordings_dir, val_files, cfg.audio, cfg.features)
    test_ds = AudioFrameDataset(recordings_dir, test_files, cfg.audio, cfg.features)

    kw: dict = dict(
        collate_fn=_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),  # pin_memory only benefits CUDA transfers
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, **kw)
    val_dl = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, **kw)
    test_dl = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, **kw)

    logger.info(
        "Dataset: train=%d val=%d test=%d frames",
        len(train_ds), len(val_ds), len(test_ds),
    )
    return train_dl, val_dl, test_dl
