# CLAUDE.md — Depression Detection from Voice (LSTM + Attention)

## Project Overview

This project demonstrates the capabilities of an **LSTM network with an attention mechanism** for detecting depression from voice recordings.
The pipeline covers the full ML lifecycle: audio preprocessing → feature extraction → overlapping frame segmentation → model training → evaluation & visualization.

**Dataset layout:**
```
recordings/          # numbered .wav files (e.g. 001.wav, 002.wav, …)
labels.csv           # columns: filename, label  (0 = healthy, 1 = depression)
```

---

## Repository Structure

```
.
├── CLAUDE.md
├── README.md
├── requirements.txt
├── recordings/              # raw .wav files
├── labels.csv
├── src/
│   ├── __init__.py
│   ├── dataset.py           # AudioDataset, frame segmentation, DataLoader factory
│   ├── features.py          # MFCC / mel-spectrogram extraction (torchaudio)
│   ├── model.py             # LSTMWithAttention nn.Module
│   ├── train.py             # training loop with early stopping
│   ├── evaluate.py          # metrics, plots, report generation
│   └── utils.py             # seed, logging helpers, config dataclass
├── configs/
│   └── default.yaml         # all hyperparameters in one place
├── scripts/
│   ├── run_train.py         # CLI entry-point: train
│   └── run_eval.py          # CLI entry-point: evaluate a saved checkpoint
├── outputs/                 # created at runtime
│   ├── checkpoints/         # best_model.pt, last_model.pt
│   ├── logs/                # TensorBoard event files
│   └── figures/             # confusion matrix, ROC, attention heatmaps
└── tests/
    ├── test_dataset.py
    ├── test_model.py
    └── test_features.py
```

---

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**`requirements.txt` (key packages):**
```
torch>=2.3.0
torchaudio>=2.3.0
numpy>=1.26
pandas>=2.2
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
tensorboard>=2.16
pyyaml>=6.0
tqdm>=4.66
pytest>=8.0
```

---

## Architecture

### 1. Audio Preprocessing & Frame Segmentation (`src/dataset.py`)

- Load `.wav` with `torchaudio.load` → resample to **16 000 Hz** mono.
- Segment each recording into **overlapping frames**:
  - `frame_length_ms = 2000`  (2 s frames)
  - `hop_length_ms   = 500`   (75 % overlap)
- Frames shorter than `frame_length_ms` are zero-padded.
- One recording → N frames; all frames inherit the recording-level label.
- **Train / val / test split** is done at the *recording* level (not frame level) to avoid data leakage.

### 2. Feature Extraction (`src/features.py`)

Each frame is converted to a sequence of feature vectors fed to the LSTM:

| Feature | Params | Output shape |
|---------|--------|--------------|
| MFCC | 40 coefficients + deltas + delta-deltas | `(T, 120)` |
| Log-mel spectrogram | 128 mels | `(T, 128)` |

Default: **MFCC 40 + Δ + ΔΔ → input size 120**.  
Set `feature_type: mel` in `configs/default.yaml` to switch.

### 3. LSTM with Attention (`src/model.py`)

```
Input  (T, input_size)
  │
  ▼
BiLSTM  ×  num_layers        hidden_size = 256 (each direction)
  │
  ▼
Additive (Bahdanau) Attention
  │   scores_t = v · tanh(W·h_t + b)
  │   α = softmax(scores)
  │   context = Σ α_t · h_t
  ▼
LayerNorm → Dropout(0.4)
  │
  ▼
Linear(hidden_size*2, num_classes=2)
  │
  ▼
Output logits
```

**Aggregation at inference:** frame-level logits are averaged per recording before the final decision.

### 4. Training (`src/train.py`)

- **Loss:** `CrossEntropyLoss` with optional class-weight balancing (`pos_weight` from label frequencies).
- **Optimizer:** `AdamW(lr=1e-3, weight_decay=1e-4)`.
- **Scheduler:** `CosineAnnealingLR` with warm restarts (`T_0=10`).
- **Early stopping:** patience = 10 epochs on validation F1.
- **Mixed precision:** `torch.autocast` (GPU only).
- Checkpoints saved to `outputs/checkpoints/`.

---

## Config (`configs/default.yaml`)

```yaml
audio:
  sample_rate: 16000
  frame_length_ms: 2000
  hop_length_ms: 500

features:
  feature_type: mfcc        # mfcc | mel
  n_mfcc: 40
  n_mels: 128
  n_fft: 512
  hop_length: 160           # in samples

model:
  hidden_size: 256
  num_layers: 2
  dropout: 0.4
  bidirectional: true

training:
  batch_size: 32
  max_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10
  seed: 42

split:
  train: 0.70
  val:   0.15
  test:  0.15
```

All values are overridable via CLI flags (e.g. `--model.hidden_size 512`).

---

## Running the Project

### Train

```bash
python scripts/run_train.py --config configs/default.yaml
```

Optional overrides:
```bash
python scripts/run_train.py \
  --config configs/default.yaml \
  --training.batch_size 64 \
  --model.num_layers 3 \
  --training.max_epochs 150
```

### Evaluate a checkpoint

```bash
python scripts/run_eval.py \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best_model.pt
```

Produces a full report in `outputs/figures/` and prints metrics to stdout.

### TensorBoard

```bash
tensorboard --logdir outputs/logs
```

---

## Evaluation & Metrics (`src/evaluate.py`)

Because depression detection is a **clinical binary classification** task with possible class imbalance, the following metrics are reported at both the *frame level* and the *recording level*:

| Metric | Rationale |
|--------|-----------|
| **Accuracy** | Baseline orientation |
| **Balanced Accuracy** | Robust to class imbalance |
| **Precision / Recall / F1** | Standard binary classification |
| **Sensitivity (Recall)** | Critical: cost of missing a depressed patient is high |
| **Specificity** | Cost of false alarms |
| **AUC-ROC** | Threshold-independent discriminability |
| **AUC-PR** | Better than ROC under class imbalance |
| **MCC** | Single balanced scalar metric |
| **Cohen's κ** | Agreement corrected for chance |

### Visualisations generated automatically

- `confusion_matrix.png` — normalised confusion matrix (seaborn heatmap)
- `roc_curve.png` — ROC with AUC annotation
- `pr_curve.png` — Precision-Recall curve
- `attention_heatmap.png` — averaged attention weights over time for correct vs. misclassified samples (qualitative interpretability)
- `training_curves.png` — loss & F1 per epoch (train / val)
- `classification_report.txt` — full sklearn report

---

## Tests

```bash
pytest tests/ -v
```

Tests cover:
- `test_dataset.py` — frame count arithmetic, label inheritance, no leakage between splits.
- `test_model.py` — forward pass shapes, attention weights sum to 1, gradient flow.
- `test_features.py` — feature tensor shapes and finite values for edge-case inputs.

---

## Code Conventions

- **Python 3.11+**, type hints everywhere.
- Config handled via a `dataclass` (not `argparse` soup) — `src/utils.py::Config`.
- `torch.manual_seed` + `numpy` seed set from `Config.seed` for reproducibility.
- All paths relative to project root; use `pathlib.Path` throughout.
- Logging via stdlib `logging` (not `print`), level controlled by `--verbose` flag.
- No Jupyter notebooks in `src/` — keep notebooks in a separate `notebooks/` dir for exploration only.

---

## Key Design Decisions

1. **Recording-level split** prevents the model from seeing different frames of the same recording in both train and test — this is the most common source of inflated accuracy in audio ML.
2. **Bidirectional LSTM** captures both past and future context within each 2-second frame.
3. **Bahdanau attention** allows the model to focus on the most diagnostically relevant segments and provides a lightweight interpretability signal (the attention heatmap).
4. **Frame-level → recording-level aggregation** by mean-pooling logits before the final decision makes the prediction robust to outlier frames.
5. **Class-weighted loss** addresses the typical imbalance in clinical datasets (fewer depressed recordings).
