# Traffic Incident Detection via LSTM Forecasting Error
 
**Mikaela Minko | University of Oregon – DSCI 410 Deep Learning | Spring 2026**
 
---
 
## Introduction
 
Traffic incidents such as accidents, sudden congestion, and road blockages disrupt flow and can significantly delay emergency response. Detecting these incidents quickly and automatically has real-world impact — but building a supervised classifier is difficult because reliable, labeled incident data is scarce.
 
This project takes an unsupervised approach: instead of learning what an incident *looks like*, an LSTM is trained to learn what **normal** traffic looks like. When the model encounters something unusual, its prediction error spikes. That forecasting error is used directly as an anomaly signal — no incident labels required.
 
The core idea is: *if the model has deeply learned normal traffic patterns, large residuals are evidence that something abnormal is happening.*
 
---
 
## Dataset
 
**Source:** [UCI Machine Learning Repository — Traffic Flow Dataset](https://archive.ics.uci.edu/ml/datasets/METR-LA)
 
The dataset contains traffic volume measurements collected from **36 highway sensors** along 2 major highways in Northern Virginia. Observations are recorded every 15 minutes. It is distributed as a MATLAB `.mat` file containing sparse matrix representations, loaded with `scipy.io.loadmat`.
 
**Train/test split:** predefined in the dataset
- Training: **1,261 timesteps** × 36 sensors = 45,396 samples
- Test: **840 timesteps** × 36 sensors = 30,240 samples
**Feature engineering (48 features per observation):**
 
| Feature Group | Features | Count |
|---|---|---|
| Historical traffic measurements (current + 9 lags) | `traffic_t-1` … `traffic_t-10` | 10 |
| Day-of-week one-hot | `weekday_0` … `weekday_6` | 7 |
| Hour-of-day one-hot | `hour_0` … `hour_23` | 24 |
| Road direction one-hot | `dir_N/S/E/W` | 4 |
| Number of lanes (scaled) | `num_lanes` | 1 |
| Road identifier one-hot | `road_name_1/2` | 2 |
| **Total** | | **48** |
 
Target values (normalized traffic volume) are scaled per-sensor using `MinMaxScaler`. Feature construction and all preprocessing is handled in `dataset.py`.
 
**Data location on Talapas:**
```
/gpfs/home/mminko/MikaelaM/traffic-anomaly-detection/data/traffic_dataset.mat
```
 
---
 
## Model
 
The model is a **two-layer LSTM** followed by two fully-connected layers, implemented in `models.py`.
 
```
Input (48 features per sensor observation)
  └─ LSTM × 2 layers (hidden=64, dropout=0.2)
       └─ Linear(64 → 32) + ReLU
            └─ Linear(32 → 1) → predicted normalised traffic volume
```
 
- **Total parameters:** ~45,000
- **Loss function:** MSE
- **Optimizer:** Adam (lr=1e-3)
- **LR scheduler:** ReduceLROnPlateau (patience=3, factor=0.5)
- **Gradient clipping:** max_norm=1.0
  
The LSTM is well-suited for this task because traffic is a time series where temporal context matters — what happened at 8:00am affects what is normal at 8:15am. The gating mechanism allows the model to learn which historical patterns are relevant and which to ignore, producing better-calibrated predictions than a feedforward baseline, which in turn makes the residuals more meaningful as anomaly scores.
 
---
 
## Training
 
**Install dependencies:**
```bash
pip install torch numpy scipy pandas scikit-learn matplotlib seaborn jupyter
```
 
**Run training:**
```bash
python train_model.py --data /gpfs/home/mminko/MikaelaM/traffic-anomaly-detection/data/traffic_dataset.mat
```
 
Optional flags:
 
| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 512 | Mini-batch size |
| `--lr` | 1e-3 | Adam learning rate |
| `--hidden` | 64 | LSTM hidden size |
| `--layers` | 2 | Number of LSTM layers |
| `--dropout` | 0.2 | Dropout between layers |
| `--out_dir` | results/ | Output directory |
 
Training saves:
- `results/trained_model.pth` — model weights
- `results/train_losses.npy` — per-epoch train MSE
- `results/val_losses.npy` — per-epoch validation MSE
**Trained model weights on Talapas:**
```
/gpfs/home/mminko/MikaelaM/traffic-anomaly-detection/results/trained_model.pth
```
 
---
 
## Results
 
### Training Convergence
 
| Metric | Value |
|--------|-------|
| Final train MSE | 0.00418 |
| Final validation MSE | 0.00575 |
| Train/validation ratio | ~1.37 |
 
The train-to-validation ratio of 1.37 indicates minimal overfitting. Both loss curves converged steadily across 50 epochs.
 
### Forecasting Performance
 
| Metric | Value |
|--------|-------|
| Overall test RMSE (normalised) | 0.0758 |
| Best per-sensor RMSE | 0.0466 |
| Worst per-sensor RMSE | 0.1313 |
| Mean per-sensor RMSE | 0.0732 |
 
### Anomaly Detection
 
The anomaly threshold was set at the **99th percentile of training residuals** (threshold = 0.2101). Any test observation with a prediction error above this threshold is flagged as anomalous.
 
| Method | Flagged | Rate |
|--------|---------|------|
| LSTM (99th pct threshold) | — | **2.2%** |
| Z-score baseline (3σ) | — | 0.1% |
 
The LSTM flagged 22× more sensor-timestep pairs than the z-score baseline. Crucially, anomalies clustered during periods of rapid traffic change rather than being randomly distributed — evidence that the model is responding to real structural shifts, not noise.
 
Visualizations of the loss curves, predicted vs. actual traffic, anomaly heat map, and per-sensor RMSE are produced in `notebooks/Evaluation.ipynb`.
 
---
 
## Limitations
 
**No ground-truth incident labels.** Without labeled incidents, there is no way to compute precision or recall. The 2.2% flagging rate is plausible for real-world traffic anomalies, but cannot be formally validated against known events.
 
**Sequence length of 1.** The current model processes each timestep independently with no explicit sequence context passed between observations. This limits the model's ability to capture rising trends or slow-building congestion patterns.
 
**Conservative z-score baseline.** The baseline uses the global mean and standard deviation across all sensors, which is a weak comparison. An hour-adjusted, per-sensor baseline would be more informative.
 
**Static threshold.** A single global threshold does not account for the fact that some sensors are inherently noisier than others. Per-sensor thresholds would likely reduce false positives on high-variance sensors.
 
### Future Work
 
- Increase sequence length to 10–12 timesteps to capture temporal trends
- Use hour-adjusted, per-sensor anomaly thresholds
- Incorporate the sensor adjacency matrix to model spatial relationships between nearby sensors
- Explore multi-horizon forecasting to detect anomalies earlier
---
 
## Repository Structure
 
```
traffic-anomaly-detection/
├── README.md
├── dataset.py              # data loading, feature engineering, train/test split
├── models.py               # TrafficLSTM architecture
├── train_model.py          # training script
├── notebooks/
│   ├── Data_Demo.ipynb     # dataset exploration and feature visualisation
│   └── Evaluation.ipynb    # model evaluation, anomaly detection, all figures
├── results/                # trained_model.pth saved here after training
└── data/                   # traffic_dataset.mat placed here
```
 
