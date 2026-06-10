#train_model.py

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from dataset import load_traffic_data
from models  import TrafficLSTM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/traffic_dataset.mat")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=512)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--out_dir",    default="results")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("Loading dataset...")
    (X_tr, Y_tr, X_te, Y_te,
     y_scaler, Y_train_scaled, Y_test_scaled,
     sensor_cols, n_train, n_test, n_sensors, n_feats, adj_matrix
    ) = load_traffic_data(args.data)

    print(f"  n_features      : {n_feats}")
    print(f"  n_sensors       : {n_sensors}")
    print(f"  train samples   : {len(Y_tr):,}")
    print(f"  test  samples   : {len(Y_te):,}")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model = TrafficLSTM(input_size=n_feats, hidden_size=64, num_layers=2, dropout=0.2)
    print(f"\nModel:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 3. Setup ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    model.to(device)
    X_tr = X_tr.to(device)
    Y_tr = Y_tr.to(device)
    X_te = X_te.to(device)
    Y_te = Y_te.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )

    loader = DataLoader(TensorDataset(X_tr, Y_tr),
                        batch_size=args.batch_size, shuffle=True)

    train_losses, val_losses = [], []

    # ── 4. Training loop ──────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, Y_batch in loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, Y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)

        train_loss = epoch_loss / len(Y_tr)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_te), Y_te).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}")

    print("\nTraining complete.")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(args.out_dir, "trained_model.pth"))
    np.save(os.path.join(args.out_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(args.out_dir, "val_losses.npy"),   np.array(val_losses))
    print(f"Saved to {args.out_dir}/")


if __name__ == "__main__":
    main()

