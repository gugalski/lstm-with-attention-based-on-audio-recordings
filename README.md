# LSTM with attention based for audio analyse

Binary classification of depression from voice recordings.  
Pipeline: raw `.wav` → MFCC features → overlapping frames → LSTM + Bahdanau attention → prediction.

> Model outputs are not a clinical diagnosis.

---

## Architecture

```
Input (T, 120)  — MFCC 40 + Δ + ΔΔ
  │
LSTM × 2        — hidden 256
  │
Bahdanau Attention
  │
LayerNorm → Dropout(0.4) → Linear → 2 logits
  │
mean-pool over frames → final prediction
```

---

## Setup

```bash
brew install python          # if needed
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Device selected automatically: **CUDA → MPS (Apple Silicon) → CPU**.

---

## Data

Place `.wav` files in `recordings/` and create `labels.csv`:

```csv
filename,label
001.wav,0
002.wav,1
```

`0` = healthy, `1` = depression. See `labels.csv.example`.

---

## Usage

**Train:**
```bash
python scripts/run_train.py --config configs/default.yaml
```

**Evaluate:**
```bash
python scripts/run_eval.py \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best_model.pt
```

**TensorBoard:**
```bash
tensorboard --logdir outputs/logs
```

Plots and metrics are saved to `outputs/figures/`.

---

## Configuration

`configs/default.yaml` — all hyperparameters in one place:

| Key | Default | Description |
|-----|---------|-------------|
| `audio.sample_rate` | 16000 | Hz |
| `audio.frame_length_ms` | 2000 | Frame duration |
| `audio.hop_length_ms` | 500 | Frame stride (75% overlap) |
| `features.feature_type` | `mfcc` | `mfcc` or `mel` |
| `features.n_mfcc` | 40 | Coefficients (×3 with Δ+ΔΔ = 120) |
| `model.hidden_size` | 256 | LSTM units |
| `model.num_layers` | 2 | Stacked LSTM layers |
| `model.dropout` | 0.4 | Dropout |
| `training.batch_size` | 32 | |
| `training.max_epochs` | 100 | |
| `training.learning_rate` | 0.001 | AdamW |
| `training.early_stopping_patience` | 10 | Epochs without val F1 improvement |
| `split.train/val/test` | 0.70/0.15/0.15 | Recording-level split |

---

## Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
configs/default.yaml
recordings/          # .wav files (git-ignored)
labels.csv           # git-ignored; see labels.csv.example
outputs/
  checkpoints/       # best_model.pt, last_model.pt
  figures/           # plots + classification_report.txt
  logs/              # TensorBoard
scripts/
  run_train.py
  run_eval.py
src/
  dataset.py         # frame segmentation, DataLoader
  features.py        # MFCC / log-mel
  model.py           # LSTMWithAttention
  train.py           # training loop
  evaluate.py        # metrics + plots
  utils.py           # Config, get_device, set_seed
tests/
```
