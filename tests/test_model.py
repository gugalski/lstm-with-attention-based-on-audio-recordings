from __future__ import annotations

import pytest
import torch

from src.model import LSTMWithAttention
from src.utils import ModelConfig


@pytest.fixture
def model() -> LSTMWithAttention:
    cfg = ModelConfig(hidden_size=32, num_layers=2, dropout=0.0)
    return LSTMWithAttention(input_size=120, cfg=cfg)


def test_output_shape(model: LSTMWithAttention) -> None:
    x = torch.randn(4, 50, 120)
    logits, weights = model(x)
    assert logits.shape == (4, 2)
    assert weights.shape == (4, 50)


def test_attention_weights_sum_to_one(model: LSTMWithAttention) -> None:
    x = torch.randn(2, 30, 120)
    _, weights = model(x)
    sums = weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-5)


def test_gradient_flow(model: LSTMWithAttention) -> None:
    x = torch.randn(2, 20, 120)
    logits, _ = model(x)
    logits.sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
