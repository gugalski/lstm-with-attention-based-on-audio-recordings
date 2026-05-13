from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.dataset import make_splits
from src.utils import Config


@pytest.fixture
def tmp_labels(tmp_path: Path) -> Path:
    rows = [{"filename": f"{i:03d}.wav", "label": i % 2} for i in range(20)]
    csv = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_split_sizes(tmp_labels: Path) -> None:
    cfg = Config()
    train, val, test = make_splits(tmp_labels, cfg)
    assert len(train) + len(val) + len(test) == 20


def test_no_overlap_between_splits(tmp_labels: Path) -> None:
    cfg = Config()
    train, val, test = make_splits(tmp_labels, cfg)
    train_files = {f for f, _ in train}
    val_files = {f for f, _ in val}
    test_files = {f for f, _ in test}
    assert train_files.isdisjoint(val_files)
    assert train_files.isdisjoint(test_files)
    assert val_files.isdisjoint(test_files)


def test_labels_inherited(tmp_labels: Path) -> None:
    cfg = Config()
    train, val, test = make_splits(tmp_labels, cfg)
    df = pd.read_csv(tmp_labels)
    label_map = dict(zip(df["filename"], df["label"]))
    for filename, label in train + val + test:
        assert label_map[filename] == label


def test_split_ratios_approximate(tmp_labels: Path) -> None:
    cfg = Config()
    train, val, _ = make_splits(tmp_labels, cfg)
    assert len(train) == pytest.approx(14, abs=1)  # 70% of 20
    assert len(val) == pytest.approx(3, abs=1)     # 15% of 20
