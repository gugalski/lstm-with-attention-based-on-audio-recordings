from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .model import LSTMWithAttention
from .utils import Config, get_device

logger = logging.getLogger(__name__)


def _run_epoch(
    model: LSTMWithAttention,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: "torch.cuda.amp.GradScaler | None",
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits, _ = model(x)
                loss = criterion(logits, y)

            if training and optimizer is not None:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * y.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


def train(
    model: LSTMWithAttention,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    output_dir: Path,
) -> None:
    device = get_device()
    logger.info("Using device: %s", device)
    model = model.to(device)

    labels = [int(y) for _, y in train_loader.dataset]
    counts = torch.bincount(torch.tensor(labels).long())
    weight = (counts.sum() / (len(counts) * counts.float())).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    best_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, cfg.training.max_epochs + 1):
        train_loss, train_f1 = _run_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_f1 = _run_epoch(model, val_loader, criterion, None, device, None)
        scheduler.step()

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("f1", {"train": train_f1, "val": val_f1}, epoch)

        logger.info(
            "Epoch %3d | train loss %.4f f1 %.4f | val loss %.4f f1 %.4f",
            epoch, train_loss, train_f1, val_loss, val_f1,
        )

        torch.save(model.state_dict(), ckpt_dir / "last_model.pt")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            logger.info("  -> new best val F1: %.4f", best_f1)
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    writer.close()
    logger.info("Training complete. Best val F1: %.4f", best_f1)
