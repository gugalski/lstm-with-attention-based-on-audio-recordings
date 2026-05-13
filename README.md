# Depression Detection from Voice — LSTM + Attention

A research-grade pipeline for detecting depression from voice recordings using a bidirectional LSTM with Bahdanau (additive) attention. The project covers the complete machine-learning lifecycle: raw audio → feature extraction → overlapping frame segmentation → model training with early stopping → clinical-grade evaluation and interpretability plots.

> **Disclaimer.** This is a demonstration project. Model outputs are not a clinical diagnosis and should not be used as one.

---

## Table of Contents

1. [Background](#background)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Environment Setup](#environment-setup)
5. [Data Preparation](#data-preparation)
6. [Audio Preprocessing & Feature Extraction](#audio-preprocessing--feature-extraction)
7. [Model Details](#model-details)
8. [Training](#training)
9. [Evaluation & Metrics](#evaluation--metrics)
10. [Configuration Reference](#configuration-reference)
11. [Running the Project](#running-the-project)
12. [Tests](#tests)
13. [Key Design Decisions](#key-design-decisions)
14. [Extending the Project](#extending-the-project)

---

## Background

Depression is one of the most prevalent mental health conditions worldwide, yet it is chronically under-diagnosed. Voice and speech carry rich acoustic cues — changes in pitch, rhythm, pausing behaviour, and articulation — that correlate with depressive episodes. Automated analysis offers a low-cost, non-invasive screening complement to clinical interviews.

This project implements a supervised binary classifier (`0` = healthy, `1` = depression) that operates directly on raw `.wav` recordings. The model processes each recording as a sequence of short overlapping frames, extracts spectral features per frame, then uses a recurrent network with attention to weight the most diagnostically informative segments before making a final prediction.

---

## Architecture

### Overview

```
Raw .wav
  │
  ▼  torchaudio.load → resample to 16 kHz → mono
Overlapping 2-second frames  (hop = 500 ms → 75 % overlap)
  │
  ▼  MFCC 40 coeff + Δ + ΔΔ
Feature tensor  (T × 120)  per frame
  │
  ▼
┌─────────────────────────────────────────────┐
│  Bidirectional LSTM  ×  num_layers           │
│  hidden_size = 256 each direction            │
│  output: (batch, T, 512)                     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Bahdanau Attention                          │
│  e_t = v · tanh(W · h_t + b)                │
│  α   = softmax(e)                            │
│  ctx = Σ αₜ · hₜ          shape: (batch, 512)│
└──────────────────┬──────────────────────────┘
                   │
                   ▼
         LayerNorm → Dropout(0.4)
                   │
                   ▼
         Linear(512 → 2)  logits
                   │
                   ▼
      (inference) mean-pool over frames
                   │
                   ▼
              Class prediction
```

### Why bidirectional LSTM?

Within a 2-second frame the model needs to relate late-frame phonetic evidence back to earlier segments (e.g., a trailing silence that contextualises an earlier hesitation). A forward-only LSTM cannot do this; a bidirectional one can.

### Why Bahdanau attention?

A simple last-hidden-state summary discards temporal structure. Bahdanau attention produces a weighted sum over all time steps, letting the model emphasise the most diagnostically relevant portions of each frame. As a side-effect, the learned weight vector `α` is directly interpretable — it is plotted as a heatmap after evaluation.

---

## Repository Structure

```
.
├── configs/
│   └── default.yaml         # single source of truth for all hyperparameters
├── notebooks/               # scratch exploration; never imported by src/
├── outputs/                 # created at runtime (contents git-ignored)
│   ├── checkpoints/         # best_model.pt  (highest val F1),  last_model.pt
│   ├── figures/             # PNG plots + classification_report.txt
│   └── logs/                # TensorBoard event files
├── recordings/              # raw .wav files (git-ignored)
├── scripts/
│   ├── run_train.py         # CLI entry-point: train
│   └── run_eval.py          # CLI entry-point: evaluate a saved checkpoint
├── src/
│   ├── __init__.py
│   ├── dataset.py           # AudioFrameDataset, make_splits, make_dataloaders
│   ├── evaluate.py          # metrics computation + all plots
│   ├── features.py          # MFCC / log-mel extraction (torchaudio)
│   ├── model.py             # BahdanauAttention, LSTMWithAttention
│   ├── train.py             # training loop, early stopping, TensorBoard writer
│   └── utils.py             # Config dataclass, set_seed, get_logger
├── tests/
│   ├── test_dataset.py      # split arithmetic, label inheritance, no leakage
│   ├── test_features.py     # tensor shapes, finite values, edge-case inputs
│   └── test_model.py        # forward shapes, attention sum = 1, gradients
├── labels.csv               # your actual labels (git-ignored)
├── labels.csv.example       # example of the required format
└── requirements.txt
```

---

## Environment Setup

**Python 3.11 or later is required.**

```bash
# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

**Key dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.3 | Model, training, autocast |
| `torchaudio` | ≥ 2.3 | Audio I/O, resampling, MFCC transforms |
| `numpy` | ≥ 1.26 | Array ops |
| `pandas` | ≥ 2.2 | Labels CSV handling |
| `scikit-learn` | ≥ 1.4 | Metrics |
| `matplotlib` / `seaborn` | ≥ 3.8 / 0.13 | Plots |
| `tensorboard` | ≥ 2.16 | Training curves |
| `pyyaml` | ≥ 6.0 | Config loading |
| `pytest` | ≥ 8.0 | Test runner |

GPU training is automatic when CUDA is available. Mixed precision (`torch.autocast`) is enabled on GPU only.

---

## Data Preparation

### Directory layout

```
recordings/
    001.wav
    002.wav
    ...
labels.csv
```

### `labels.csv` format

```csv
filename,label
001.wav,0
002.wav,1
003.wav,0
```

- `filename` — base name of the `.wav` file inside `recordings/`
- `label` — `0` for healthy, `1` for depression
- The file must have a header row

A sample file is provided as `labels.csv.example`.

### Audio requirements

- Any sample rate — recordings are automatically resampled to 16 000 Hz
- Mono or stereo — stereo channels are averaged to mono
- Any duration — recordings are sliced into overlapping 2-second frames; very short recordings produce a single zero-padded frame

---

## Audio Preprocessing & Feature Extraction

### Frame segmentation (`src/dataset.py`)

Each recording is split into overlapping frames before feature extraction:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `frame_length_ms` | 2000 ms | Each frame is 2 seconds of audio |
| `hop_length_ms` | 500 ms | Frames advance by 0.5 s → 75 % overlap |

A 30-second recording at 16 kHz produces roughly `⌊(30 000 − 2000) / 500⌋ + 1 = 57` frames. Frames that are shorter than `frame_length_ms` (typically the last one) are zero-padded on the right.

All frames from one recording share the same recording-level label. The train/val/test split is performed at the **recording** level (before frame generation) to prevent any frame from the same recording appearing in more than one split.

### MFCC features (default, `src/features.py`)

Each frame is transformed into a sequence of feature vectors:

```
frame waveform  (1, 32000)
  → MFCC transform  → (40, T)        40 cepstral coefficients
  → compute_deltas  → (40, T)        first-order deltas
  → compute_deltas² → (40, T)        second-order deltas
  → concatenate     → (120, T)
  → transpose       → (T, 120)       fed to the LSTM as a sequence
```

### Log-mel spectrogram (alternative)

Set `feature_type: mel` in `configs/default.yaml` to use log-mel spectrograms instead:

```
frame waveform  (1, 32000)
  → MelSpectrogram  → (128, T)       128 mel bands
  → log(x + 1e-9)  → (128, T)
  → transpose       → (T, 128)
```

The model `input_size` is inferred automatically from the first batch, so switching feature types requires no code change.

---

## Model Details

### `BahdanauAttention` (`src/model.py`)

```python
e_t = v · tanh(W · h_t + b)     # scalar energy per time step
α   = softmax(e)                  # attention weights, sum to 1
ctx = Σ αₜ · hₜ                  # weighted context vector
```

`W` and `v` are learned linear projections with no bias (following the original Bahdanau formulation). The resulting weight vector `α` is returned alongside logits and visualised during evaluation.

### `LSTMWithAttention` (`src/model.py`)

| Layer | Output shape | Notes |
|-------|-------------|-------|
| Input | `(B, T, 120)` | B = batch size, T = time steps |
| BiLSTM × 2 | `(B, T, 512)` | 256 per direction; inter-layer dropout |
| Attention | `(B, 512)` | context vector + weights `(B, T)` |
| LayerNorm | `(B, 512)` | stabilises training |
| Dropout(0.4) | `(B, 512)` | regularisation |
| Linear | `(B, 2)` | raw logits |

Total trainable parameters with defaults: ~3.5 M.

---

## Training

### Loss

`CrossEntropyLoss` with per-class weights computed from training label frequencies:

```
weight_c = N_total / (num_classes × N_c)
```

This automatically compensates for class imbalance without manual tuning.

### Optimiser & scheduler

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Optimiser | `AdamW(lr=1e-3, wd=1e-4)` | Weight decay decoupled from gradient update |
| Scheduler | `CosineAnnealingWarmRestarts(T_0=10)` | Periodic restarts escape local minima |

### Early stopping

Training halts if validation macro-F1 does not improve for `early_stopping_patience` consecutive epochs (default: 10). The checkpoint with the best validation F1 is saved as `best_model.pt`; the final epoch state is saved as `last_model.pt`.

### Mixed precision

On CUDA devices, `torch.autocast` reduces memory usage and increases throughput at no cost to accuracy.

### TensorBoard logging

Loss and F1 are logged per epoch under the `loss` and `f1` tags. Launch the viewer with:

```bash
tensorboard --logdir outputs/logs
```

---

## Evaluation & Metrics

Metrics are computed on the held-out test split. Because depression detection is a **clinical binary classification** task with possible class imbalance, a broad set of metrics is reported:

| Metric | Why it matters |
|--------|---------------|
| **Accuracy** | Baseline orientation — misleading under imbalance |
| **Balanced Accuracy** | Mean recall per class — robust to imbalance |
| **F1 (macro)** | Harmonic mean of precision and recall across classes |
| **Sensitivity (Recall, class 1)** | The cost of missing a depressed patient is high |
| **Specificity (Recall, class 0)** | The cost of false alarms (unnecessary clinical referral) |
| **AUC-ROC** | Threshold-independent discriminability |
| **AUC-PR** | More informative than ROC under severe class imbalance |
| **MCC** | Single balanced scalar; unaffected by class size |
| **Cohen's κ** | Agreement corrected for chance |

### Generated figures

| File | Content |
|------|---------|
| `confusion_matrix.png` | Normalised heatmap (row = true, col = predicted) |
| `roc_curve.png` | ROC curve with AUC annotation |
| `pr_curve.png` | Precision-Recall curve with AP annotation |
| `attention_heatmap.png` | Mean attention weights for correctly vs. misclassified samples |
| `classification_report.txt` | Full `sklearn` classification report |

---

## Configuration Reference

All hyperparameters are in `configs/default.yaml`. The full schema:

```yaml
audio:
  sample_rate: 16000          # Hz — recordings are resampled to this
  frame_length_ms: 2000       # ms — length of each analysis frame
  hop_length_ms: 500          # ms — stride between frame starts (75% overlap)

features:
  feature_type: mfcc          # "mfcc" | "mel"
  n_mfcc: 40                  # number of MFCC coefficients (×3 with Δ+ΔΔ → 120 total)
  n_mels: 128                 # mel bands (used for both MFCC filterbank and mel mode)
  n_fft: 512                  # FFT window size in samples
  hop_length: 160             # STFT hop in samples (~10 ms at 16 kHz)

model:
  hidden_size: 256            # LSTM units per direction
  num_layers: 2               # stacked LSTM layers
  dropout: 0.4                # applied between LSTM layers and before classifier
  bidirectional: true

training:
  batch_size: 32              # frames per batch
  max_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10 # epochs of no val-F1 improvement before stopping
  seed: 42

split:
  train: 0.70                 # recording-level fractions (must sum to 1)
  val:   0.15
  test:  0.15
```

All values can be overridden directly via CLI flags — see [Running the Project](#running-the-project).

---

## Running the Project

### 1. Prepare data

```
recordings/
    001.wav
    002.wav
    ...
labels.csv     # see labels.csv.example
```

### 2. Train

```bash
python scripts/run_train.py --config configs/default.yaml
```

With optional overrides:

```bash
python scripts/run_train.py \
  --config configs/default.yaml \
  --recordings recordings \
  --labels labels.csv \
  --output-dir outputs \
  --verbose
```

Progress is printed to stdout and logged to TensorBoard. Checkpoints are written to `outputs/checkpoints/`.

### 3. Monitor training

```bash
tensorboard --logdir outputs/logs
```

Open `http://localhost:6006` in your browser.

### 4. Evaluate

```bash
python scripts/run_eval.py \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best_model.pt
```

Metrics are printed to stdout. All plots and the classification report are saved to `outputs/figures/`.

### 5. Experiment with features

Switch to log-mel spectrograms by editing `configs/default.yaml`:

```yaml
features:
  feature_type: mel
```

No other changes are needed — `input_size` is inferred automatically.

---

## Tests

```bash
pytest tests/ -v
```

| Test file | Coverage |
|-----------|---------|
| `test_model.py` | Forward pass output shapes; attention weights sum to 1 across all time steps; gradients flow to every parameter |
| `test_features.py` | MFCC output shape is `(T, 120)`; mel output shape is `(T, 128)`; all values finite; silent audio handled; unknown `feature_type` raises `ValueError` |
| `test_dataset.py` | Train + val + test sizes sum to total; no filename appears in more than one split; every sample inherits its recording's label; split ratios are approximately correct |

Tests use only CPU and synthetic/fixture data — no real recordings required.

---

## Key Design Decisions

### Recording-level split (not frame-level)

The most common source of inflated accuracy in audio ML is splitting at the frame level: the model sees frame 3 of recording `007.wav` in training and frame 7 of the same recording at test time, leading to near-perfect validation scores that do not generalise. This project always splits at the **recording** level — all frames from one recording belong to exactly one partition.

### 75 % frame overlap

A 500 ms hop on 2-second frames means adjacent frames share 1.5 seconds of audio. This serves two purposes: it multiplies the number of training examples (important for small clinical datasets), and it smooths the per-recording prediction because the averaged logits blend overlapping temporal contexts.

### Mean-pooled logits at inference

Rather than taking the majority vote of frame-level class predictions, raw logits are averaged before the softmax. Averaging in logit space is more numerically stable and retains calibration information from every frame, making the final prediction more robust to occasional outlier frames.

### Class-weighted cross-entropy

Clinical datasets are rarely balanced. Computing per-class weights from training label frequencies and passing them to `CrossEntropyLoss` removes the need to manually tune loss weights or oversample the minority class.

### Bahdanau vs. dot-product attention

Dot-product (Luong) attention scales poorly when the hidden dimension is large because the dot products grow in magnitude, saturating softmax. Bahdanau attention introduces a learned projection that keeps scores in a stable range regardless of hidden size, which is important here because the BiLSTM hidden dim is 512.

---

## Extending the Project

| Goal | Where to change |
|------|----------------|
| Add a new feature type (e.g., chroma, spectral contrast) | Add a new branch in `src/features.py::extract_features` |
| Try a Transformer encoder instead of LSTM | Replace `self.lstm` in `LSTMWithAttention` with `nn.TransformerEncoder`; keep the attention and classifier unchanged |
| Multi-class severity labels | Change `num_classes` in `LSTMWithAttention`; update the loss and metrics accordingly |
| Cross-validation | Replace `make_splits` in `src/dataset.py` with a k-fold loop at the recording level |
| Hyperparameter search | Wrap `scripts/run_train.py` in Optuna or Ray Tune; all knobs are in `Config` |
| Export for inference | Call `torch.jit.script(model)` or `torch.onnx.export(model, ...)` after loading a checkpoint |
