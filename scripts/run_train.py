#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import make_dataloaders
from src.model import LSTMWithAttention
from src.train import train
from src.utils import Config, get_logger, set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM+Attention depression detector")
    p.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    p.add_argument("--recordings", default="recordings", type=Path)
    p.add_argument("--labels", default="labels.csv", type=Path)
    p.add_argument("--output-dir", default="outputs", type=Path)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = Config.from_yaml(ROOT / args.config)
    logger = get_logger("run_train", verbose=args.verbose)
    set_seed(cfg.training.seed)

    recordings_dir = ROOT / args.recordings
    labels_csv = ROOT / args.labels
    output_dir = ROOT / args.output_dir

    train_dl, val_dl, _ = make_dataloaders(recordings_dir, labels_csv, cfg)

    x, _ = next(iter(train_dl))
    input_size = x.shape[-1]
    logger.info("Input feature size: %d", input_size)

    model = LSTMWithAttention(input_size=input_size, cfg=cfg.model)
    train(model, train_dl, val_dl, cfg, output_dir)


if __name__ == "__main__":
    main()
