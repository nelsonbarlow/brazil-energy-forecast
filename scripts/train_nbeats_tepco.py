#!/usr/bin/env python3
"""
Train N-BEATS on TEPCO (Tokyo) data and compare with zero-shot Chronos-2.

Locally-trained baseline for the universality experiment: does a model
trained for 5 years on the Japanese grid beat or match a zero-shot FM?
Best config carried over from the Brazilian N-BEATS sweep (input=168h, lr=5e-4).

Train: 2019-01-01 – 2023-12-31  (~43,800 h, 5 years)
Test:  2024-01-01 – 2024-12-31  (8,784 h, leap year)

Usage:
    python scripts/train_nbeats_tepco.py
    python scripts/train_nbeats_tepco.py --epochs 300 --device cpu
"""

import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, root_mean_squared_error,
    mean_absolute_percentage_error, r2_score,
)
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results')

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate(actual, predicted, model_name):
    actual = np.asarray(actual).squeeze()
    predicted = np.asarray(predicted).squeeze()
    naive_err = mean_absolute_error(actual[24:], actual[:-24])
    metrics = {
        'MAE (MW)':  mean_absolute_error(actual, predicted),
        'RMSE (MW)': root_mean_squared_error(actual, predicted),
        'MAPE':      mean_absolute_percentage_error(actual, predicted) * 100,
        'MASE':      mean_absolute_error(actual, predicted) / naive_err,
        'RMSSE':     np.sqrt(mean_squared_error(actual, predicted) /
                     mean_squared_error(actual[24:], actual[:-24])),
        'R2':        r2_score(actual, predicted),
    }
    print(f'\n{"─"*60}')
    print(f'  {model_name}')
    print(f'{"─"*60}')
    for k, v in metrics.items():
        print(f'  {k:>12s}: {v:.2f}{"%" if k == "MAPE" else ""}')
    return metrics

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tepco():
    pattern = os.path.join(PROCESSED_DIR, 'tepco_hourly_load_*.parquet')
    files = glob.glob(pattern)
    if not files:
        pattern = os.path.join(PROCESSED_DIR, 'tepco_hourly_load_*.csv')
        files = glob.glob(pattern)
    if not files:
        print('No TEPCO data. Run: python scripts/download_tepco.py --start 2019 --end 2024')
        sys.exit(1)

    frames = [pd.read_parquet(f) if f.endswith('.parquet') else pd.read_csv(f)
              for f in sorted(files)]
    df = pd.concat(frames, ignore_index=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.drop_duplicates('datetime').sort_values('datetime').reset_index(drop=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Train N-BEATS on TEPCO data; compare with zero-shot Chronos-2')
    parser.add_argument('--horizon', type=int, default=24)
    # Best config from Brazilian sweep: input_length=168, lr=5e-4
    parser.add_argument('--input-length', type=int, default=168)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-stacks', type=int, default=30)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--layer-widths', type=int, default=256)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--val-days', type=int, default=60)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    print(f'Grid:    TEPCO (Tokyo, Japan)')
    print(f'Train:   2019–2023  |  Test: 2024')
    print(f'Horizon: {args.horizon}h  Input: {args.input_length}h')
    print(f'Epochs:  {args.epochs}  LR: {args.lr}  Batch: {args.batch_size}')

    from darts import TimeSeries, concatenate
    from darts.models import NBEATSModel
    from pytorch_lightning.callbacks import EarlyStopping

    df = load_tepco()

    series = TimeSeries.from_dataframe(
        df, time_col='datetime', value_cols='demand_mw', freq='h',
        fill_missing_dates=True,
    )
    print(f'\nTimeSeries: {len(series)} hours  '
          f'({series.start_time().date()} – {series.end_time().date()})')

    # Split: train 2019–2023, val = last 60 days of 2023, test = 2024
    test_start  = pd.Timestamp('2024-01-01')
    test_series  = series.slice(test_start, pd.Timestamp('2024-12-31 23:00:00'))
    pre_test     = series.slice(series.start_time(), test_start - pd.Timedelta(hours=1))
    val_hours    = args.val_days * 24
    val_series   = pre_test[-val_hours:]
    train_series = pre_test[:-val_hours]

    print(f'Train:   {len(train_series)} h ({len(train_series)//24} days)')
    print(f'Val:     {len(val_series)} h ({args.val_days} days)')
    print(f'Test:    {len(test_series)} h (2024)')

    early_stopper = EarlyStopping(
        monitor='val_loss', patience=args.patience, min_delta=1e-5, mode='min',
    )

    model = NBEATSModel(
        input_chunk_length=args.input_length,
        output_chunk_length=args.horizon,
        num_stacks=args.num_stacks,
        num_blocks=1,
        num_layers=args.num_layers,
        layer_widths=args.layer_widths,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={'lr': args.lr},
        lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_kwargs={'patience': 5, 'factor': 0.5, 'min_lr': 1e-6},
        pl_trainer_kwargs={
            'callbacks': [early_stopper],
            'accelerator': 'cpu',
        },
        random_state=args.random_state,
        force_reset=True,
        save_checkpoints=True,
    )

    print(f'\n{"━"*60}')
    print(f'  Training N-BEATS on TEPCO 2019–2023...')
    print(f'{"━"*60}')
    model.fit(series=train_series, val_series=val_series, verbose=True)

    print(f'\n{"━"*60}')
    print(f'  Rolling backtest on 2024...')
    print(f'{"━"*60}')
    backtest = model.historical_forecasts(
        series=series,
        start=test_series.start_time(),
        forecast_horizon=args.horizon,
        stride=args.horizon,
        retrain=False,
        last_points_only=False,
        verbose=True,
    )
    forecast = concatenate(backtest)

    pred_np   = forecast.values().flatten()
    actual_np = test_series.values().flatten()
    min_len   = min(len(pred_np), len(actual_np))
    pred_np, actual_np = pred_np[:min_len], actual_np[:min_len]

    nbeats_metrics = evaluate(actual_np, pred_np, 'N-BEATS (trained on TEPCO 2019–2023)')

    # ------------------------------------------------------------------
    # Compare with zero-shot results from benchmark_tepco.py
    # ------------------------------------------------------------------
    all_metrics = {'N-BEATS (trained)': nbeats_metrics}

    zs_path = os.path.join(OUTPUT_DIR, f'benchmark_TEPCO_{args.horizon}h_2024.csv')
    if os.path.exists(zs_path):
        zs_df = pd.read_csv(zs_path, index_col=0)
        for name in zs_df.index:
            all_metrics[f'{name} (zero-shot)'] = zs_df.loc[name].to_dict()
    else:
        print(f'\n(Zero-shot results not found at {zs_path} — run benchmark_tepco.py first)')

    print(f'\n{"="*70}')
    print(f'  FULL COMPARISON: TEPCO Tokyo, {args.horizon}h horizon, test=2024')
    print(f'{"="*70}')
    results_df = pd.DataFrame(all_metrics).T.round(2)
    if 'MAPE' in results_df.columns:
        results_df = results_df.sort_values('MAPE')
    print(results_df.to_string())

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f'nbeats_tepco_{args.horizon}h_2024.csv')
    results_df.to_csv(csv_path)
    print(f'\nCSV: {csv_path}')


if __name__ == '__main__':
    main()
