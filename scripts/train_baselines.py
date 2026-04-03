#!/usr/bin/env python3
"""
Train local baselines (LSTM, Linear) on ONS data and compare with
zero-shot foundation model results.

Usage:
    python scripts/train_baselines.py                          # SE, 24h
    python scripts/train_baselines.py --subsystem NE           # Nordeste
    python scripts/train_baselines.py --horizon 168            # 1 week ahead
    python scripts/train_baselines.py --epochs 200 --lr 1e-3   # tuning
"""

import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, root_mean_squared_error,
    mean_absolute_percentage_error, r2_score,
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results')

# ---------------------------------------------------------------------------
# Metrics (same as benchmark.py)
# ---------------------------------------------------------------------------
def _naive_forecast(actual, seasonality=1):
    return actual[:-seasonality]

def evaluate(actual, predicted, model_name):
    actual = np.asarray(actual).squeeze()
    predicted = np.asarray(predicted).squeeze()
    metrics = {
        'MAE (MW)':  mean_absolute_error(actual, predicted),
        'RMSE (MW)': root_mean_squared_error(actual, predicted),
        'MAPE':      mean_absolute_percentage_error(actual, predicted) * 100,
        'MASE':      mean_absolute_error(actual, predicted) /
                     mean_absolute_error(actual[24:], _naive_forecast(actual, 24)),
        'RMSSE':     np.sqrt(mean_squared_error(actual, predicted) /
                     mean_squared_error(actual[24:], _naive_forecast(actual, 24))),
        'R2':        r2_score(actual, predicted),
    }
    pct = {'MAPE'}
    print(f'\n{"─"*60}')
    print(f'  {model_name}')
    print(f'{"─"*60}')
    for k, v in metrics.items():
        fmt = f'{v:.2f}%' if k in pct else f'{v:.2f}'
        print(f'  {k:>12s}: {fmt}')
    return metrics

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class LoadDataset(Dataset):
    def __init__(self, data, context_len, horizon):
        self.data = data
        self.context_len = context_len
        self.horizon = horizon
        self.n_samples = len(data) - context_len - horizon + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.context_len]
        y = self.data[idx + self.context_len : idx + self.context_len + self.horizon]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2,
                 horizon=24, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, horizon)
        self.horizon = horizon

    def forward(self, x):
        # x: (batch, seq_len) -> (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        # Take last hidden state
        return self.fc(out[:, -1, :])


class LinearForecaster(nn.Module):
    def __init__(self, context_len, horizon):
        super().__init__()
        self.fc = nn.Linear(context_len, horizon)

    def forward(self, x):
        return self.fc(x)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs, lr, device, name):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10,
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 30

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'  Epoch {epoch+1:3d}/{epochs} | '
                  f'Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr_now:.1e}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  Early stopping at epoch {epoch+1}')
                break

    model.load_state_dict(best_state)
    model = model.to(device)
    print(f'  Best val loss: {best_val_loss:.4f}')
    return model

# ---------------------------------------------------------------------------
# Rolling prediction (same protocol as benchmark.py)
# ---------------------------------------------------------------------------
def rolling_predict(model, full_data, test_start_idx, test_len, context_len,
                    horizon, mean, std, device, desc):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in tqdm(range(0, test_len, horizon), desc=desc):
            ctx = full_data[test_start_idx + i - context_len : test_start_idx + i]
            ctx_norm = (ctx - mean) / std
            x = torch.FloatTensor(ctx_norm).unsqueeze(0).to(device)
            pred_norm = model(x).squeeze(0).cpu().numpy()
            actual_horizon = min(horizon, test_len - i)
            pred = pred_norm[:actual_horizon] * std + mean
            preds.extend(pred.tolist())
    return np.array(preds[:test_len])

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(subsystem):
    pattern = os.path.join(PROCESSED_DIR, f'ons_hourly_load_{subsystem.lower()}_*.csv')
    files = glob.glob(pattern)
    if not files:
        pattern = os.path.join(PROCESSED_DIR, 'ons_hourly_load_all_*.csv')
        files = glob.glob(pattern)
    if not files:
        print(f'No data found. Run: python scripts/download_ons.py --subsystem {subsystem}')
        sys.exit(1)
    df = pd.read_csv(files[0])
    df['datetime'] = pd.to_datetime(df['datetime'])
    if 'subsystem' in df.columns and len(df['subsystem'].unique()) > 1:
        df = df[df['subsystem'] == subsystem.upper()].reset_index(drop=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train local baselines on ONS data')
    parser.add_argument('--subsystem', type=str, default='SE',
                        choices=['SE', 'S', 'NE', 'N'])
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--context-length', type=int, default=168,
                        help='Context window (default: 168 = 1 week)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--test-days', type=int, default=365)
    parser.add_argument('--val-days', type=int, default=60,
                        help='Days for validation (default: 60)')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Subsystem: {args.subsystem}')
    print(f'Device: {device}')
    print(f'Horizon: {args.horizon}h, Context: {args.context_length}h')
    print(f'Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}')

    # Load data
    df = load_data(args.subsystem)
    full_load = df['load_mw'].values.astype(np.float32)

    # Split: train | val | test
    test_hours = args.test_days * 24
    val_hours = args.val_days * 24
    test_start = len(full_load) - test_hours
    val_start = test_start - val_hours

    train_data = full_load[:val_start]
    val_data = full_load[val_start - args.context_length : test_start]
    test_actuals = full_load[test_start:]

    # Normalize using training stats
    train_mean = train_data.mean()
    train_std = train_data.std()

    train_norm = (train_data - train_mean) / train_std
    val_norm = (full_load[val_start - args.context_length : test_start] - train_mean) / train_std

    print(f'\nData split:')
    print(f'  Train: {len(train_data):,} hours ({len(train_data)//24:,} days)')
    print(f'  Val:   {val_hours:,} hours ({args.val_days} days)')
    print(f'  Test:  {test_hours:,} hours ({args.test_days} days)')
    print(f'  Mean:  {train_mean:,.0f} MW, Std: {train_std:,.0f} MW')

    # Datasets
    train_ds = LoadDataset(train_norm, args.context_length, args.horizon)
    val_ds = LoadDataset(val_norm, args.context_length, args.horizon)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    print(f'  Train samples: {len(train_ds):,}')
    print(f'  Val samples:   {len(val_ds):,}')

    all_metrics = {}

    # ------------------------------------------------------------------
    # Model 1: LSTM
    # ------------------------------------------------------------------
    print(f'\n{"━"*60}')
    print(f'  Training LSTM (hidden={args.hidden_size}, 2 layers)...')
    print(f'{"━"*60}')

    lstm = LSTMForecaster(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=2,
        horizon=args.horizon,
        dropout=0.1,
    )
    n_params = sum(p.numel() for p in lstm.parameters())
    print(f'  Parameters: {n_params:,}')

    lstm = train_model(lstm, train_loader, val_loader,
                       args.epochs, args.lr, device, 'LSTM')

    lstm_preds = rolling_predict(
        lstm, full_load, test_start, len(test_actuals),
        args.context_length, args.horizon, train_mean, train_std,
        device, 'LSTM',
    )
    all_metrics['LSTM (trained)'] = evaluate(test_actuals, lstm_preds, 'LSTM (trained)')

    # ------------------------------------------------------------------
    # Model 2: Linear
    # ------------------------------------------------------------------
    print(f'\n{"━"*60}')
    print(f'  Training Linear...')
    print(f'{"━"*60}')

    linear = LinearForecaster(args.context_length, args.horizon)
    n_params = sum(p.numel() for p in linear.parameters())
    print(f'  Parameters: {n_params:,}')

    linear = train_model(linear, train_loader, val_loader,
                         args.epochs, args.lr, device, 'Linear')

    linear_preds = rolling_predict(
        linear, full_load, test_start, len(test_actuals),
        args.context_length, args.horizon, train_mean, train_std,
        device, 'Linear',
    )
    all_metrics['Linear (trained)'] = evaluate(test_actuals, linear_preds, 'Linear (trained)')

    # ------------------------------------------------------------------
    # Load zero-shot results for comparison
    # ------------------------------------------------------------------
    zs_path = os.path.join(OUTPUT_DIR, f'benchmark_{args.subsystem}_{args.horizon}h.csv')
    if os.path.exists(zs_path):
        zs_df = pd.read_csv(zs_path, index_col=0)
        for model_name in zs_df.index:
            all_metrics[f'{model_name} (zero-shot)'] = zs_df.loc[model_name].to_dict()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f'\n{"="*70}')
    print(f'  COMPARISON: {args.subsystem} subsystem, {args.horizon}h horizon')
    print(f'  Trained models vs Zero-shot foundation models')
    print(f'{"="*70}')
    results_df = pd.DataFrame(all_metrics).T.round(2)
    # Sort by MAPE
    if 'MAPE' in results_df.columns:
        results_df = results_df.sort_values('MAPE')
    print(results_df.to_string())

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f'trained_vs_zeroshot_{args.subsystem}_{args.horizon}h.csv')
    results_df.to_csv(csv_path)
    print(f'\nCSV: {csv_path}')


if __name__ == '__main__':
    main()
