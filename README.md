# Traffic Incident Detection via LSTM Forecasting Error

**Mikaela Minko | University of Oregon – DSCI 410 Deep Learning | Spring 2026**

Unsupervised anomaly detection for highway traffic sensors using LSTM prediction
error as an anomaly signal — no incident labels required.

---

## Project Overview

Traffic incidents disrupt flow but labeled incident data is scarce. This project
trains a two-layer LSTM to learn *normal* traffic patterns across 36 highway
sensors. Large prediction residuals are then treated as anomaly scores, allowing
incident detection without any ground-truth labels.

**Key results**
- Test RMSE: **0.0758** (normalised)
- Anomaly threshold (99th pct of training residuals): **0.2101**
- Flagged **2.2%** of sensor-timestep pairs as anomalous vs. 0.1% for z-score baseline

---

## Repository Structure

```
traffic-anomaly-detection/
├── README.md
├── dataset.py          # data loading, feature engineering, train/test split
├── models.py           # TrafficLSTM architecture
├── train_model.py      # training script (CLI)
├── notebooks/
│   ├── Data_Demo.ipynb      # dataset exploration & feature visualisation
│   └── Evaluation.ipynb     # anomaly detection analysis & figures
├── results/
│   └── trained_model.pth    # saved weights (after training)
└── data/
    └── traffic_dataset.mat  # raw dataset (not tracked in git — see below)
```

---

## Setup

```bash
pip install torch numpy scipy matplotlib seaborn jupyter
```

Python 3.9+ recommended.

---

## Data

Download the Traffic Flow Dataset from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/METR-LA)
and place the `.mat` file at `data/traffic_dataset.mat`.

The file is loaded with `scipy.io.loadmat` — see `dataset.py` for expected
variable names (`volume`, `day_of_week`, `hour_of_day`, `direction`, `n_lanes`, `road_id`).

---

## Training

```bash
python train_model.py --data data/traffic_dataset.mat --epochs 50
```

Optional flags:
| Flag | Default | Description |
|------|---------|-------------|
| `--batch_size` | 512 | Mini-batch size |
| `--lr` | 1e-3 | Adam learning rate |
| `--hidden` | 64 | LSTM hidden size |
| `--layers` | 2 | Number of LSTM layers |
| `--dropout` | 0.2 | Dropout between layers |
| `--out_dir` | results/ | Where to save weights + loss curves |

Saved outputs: `results/trained_model.pth`, `results/train_losses.npy`, `results/val_losses.npy`

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `Data_Demo.ipynb` | Load the dataset, inspect feature distributions, visualise traffic patterns |
| `Evaluation.ipynb` | Load trained model, compute residuals, set anomaly threshold, generate heat maps and RMSE-per-sensor plots |

---

## Model Architecture

```
Input (48 features per sensor observation)
  └─ LSTM × 2 layers (hidden=64, dropout=0.2)
       └─ Linear(64 → 32) + ReLU
            └─ Linear(32 → 1) → predicted traffic volume
```

Total parameters: ~45,000

---

## Anomaly Detection

Residuals are computed as `|predicted - actual|` for every sensor-timestep pair
in the test set. The anomaly threshold is the **99th percentile** of training
residuals. Any test observation exceeding this threshold is flagged as anomalous.

A z-score baseline (global mean / std) is included for comparison in `Evaluation.ipynb`.
