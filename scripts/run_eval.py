#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import make_dataloaders
from src.evaluate import evaluate
from src.model import LSTMWithAttention
from src.utils import Config, get_device, get_logger, set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved LSTM+Attention checkpoint")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--recordings", default="recordings", type=Path)
    p.add_argument("--labels", default="labels.csv", type=Path)
    p.add_argument("--output-dir", default="outputs", type=Path)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = Config.from_yaml(ROOT / args.config)
    logger = get_logger("run_eval", verbose=args.verbose)
    set_seed(cfg.training.seed)

    recordings_dir = ROOT / args.recordings
    labels_csv = ROOT / args.labels
    output_dir = ROOT / args.output_dir

    _, _, test_dl = make_dataloaders(recordings_dir, labels_csv, cfg)

    x, _ = next(iter(test_dl))
    input_size = x.shape[-1]

    device = get_device()
    model = LSTMWithAttention(input_size=input_size, cfg=cfg.model)
    state = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    metrics = evaluate(model, test_dl, device, output_dir)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
