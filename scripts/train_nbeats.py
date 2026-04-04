#!/usr/bin/env python3
"""
Train N-BEATS (via Darts) on ONS load data and compare with
zero-shot foundation model results.

N-BEATS is a state-of-the-art deep learning model for time series
forecasting. This provides the proper trained DL baseline that
reviewers expect.

Usage:
    python scripts/train_nbeats.py                          # SE, 24h
    python scripts/train_nbeats.py --subsystem NE           # Nordeste
    python scripts/train_nbeats.py --epochs 100 --lr 5e-4   # tuning

Requires: pip install "darts[torch]"
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
# Metrics (same as benchmark.py)
# ---------------------------------------------------------------------------
def _naive_forecast(actual, seasonality=1):
    return actual[:-seasonality]

def evaluate(actual, predicted, model_name):
    actual = np.asarray(actual).squeeze()
    predicted = np.asarray(predicted).squeeze()
    naive_err = mean_absolute_error(actual[24:], _naive_forecast(actual, 24))
    metrics = {
        'MAE (MW)':  mean_absolute_error(actual, predicted),
        'RMSE (MW)': root_mean_squared_error(actual, predicted),
        'MAPE':      mean_absolute_percentage_error(actual, predicted) * 100,
        'MASE':      mean_absolute_error(actual, predicted) / naive_err,
        'RMSSE':     np.sqrt(mean_squared_error(actual, predicted) /
                     mean_squared_error(actual[24:], _naive_forecast(actual, 24))),
        'R2':        r2_score(actual, predicted),
    }
    pct = {'MAPE'}
    print(f'\n{"─"*60}')
    print(f'  {model_name}')
    print(f'{"─"*60}')
    for k, v in metrics.items():
        print(f'  {k:>12s}: {v:.2f}{"%" if k in pct else ""}')
    return metrics

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
        print(f'No data. Run: python scripts/download_ons.py --subsystem {subsystem}')
        sys.exit(1)
    df = pd.read_csv(files[0])
    df['datetime'] = pd.to_datetime(df['datetime'])
    if 'subsystem' in df.columns and len(df['subsystem'].unique()) > 1:
        df = df[df['subsystem'] == subsystem.upper()].reset_index(drop=True)
    return df.sort_values('datetime').reset_index(drop=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train N-BEATS on ONS load data')
    parser.add_argument('--subsystem', type=str, default='SE',
                        choices=['SE', 'S', 'NE', 'N'])
    parser.add_argument('--horizon', type=int, default=24,
                        help='Forecast horizon in hours (default: 24)')
    parser.add_argument('--input-length', type=int, default=168,
                        help='Input chunk length in hours (default: 168 = 1 week)')
    parser.add_argument('--test-days', type=int, default=365)
    parser.add_argument('--val-days', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-stacks', type=int, default=30)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--layer-widths', type=int, default=256)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    print(f'Subsystem: {args.subsystem}')
    print(f'Horizon: {args.horizon}h, Input: {args.input_length}h')
    print(f'Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}')
    print(f'N-BEATS: {args.num_stacks} stacks, {args.num_layers} layers, {args.layer_widths} width')

    # ------------------------------------------------------------------
    # Load data and create Darts TimeSeries
    # ------------------------------------------------------------------
    from darts import TimeSeries, concatenate
    from darts.models import NBEATSModel
    from pytorch_lightning.callbacks import EarlyStopping

    df = load_data(args.subsystem)

    # Darts needs a regular frequency — fill any gaps
    series = TimeSeries.from_dataframe(
        df, time_col='datetime', value_cols='load_mw', freq='h',
        fill_missing_dates=True,
    )
    print(f'\nTimeSeries length: {len(series)} hours')

    # ------------------------------------------------------------------
    # Split: train | val | test
    # ------------------------------------------------------------------
    test_hours = args.test_days * 24
    val_hours = args.val_days * 24

    test_series = series[-test_hours:]
    pre_test = series[:-test_hours]
    val_series = pre_test[-val_hours:]
    train_series = pre_test[:-val_hours]

    print(f'Train: {len(train_series)} hours ({len(train_series)//24} days)')
    print(f'Val:   {len(val_series)} hours ({args.val_days} days)')
    print(f'Test:  {len(test_series)} hours ({args.test_days} days)')

    # ------------------------------------------------------------------
    # Configure N-BEATS
    # ------------------------------------------------------------------
    early_stopper = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        min_delta=1e-5,
        mode='min',
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

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f'\n{"━"*60}')
    print(f'  Training N-BEATS...')
    print(f'{"━"*60}')

    model.fit(
        series=train_series,
        val_series=val_series,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Rolling backtest on test set (same protocol as benchmark.py)
    # ------------------------------------------------------------------
    print(f'\n{"━"*60}')
    print(f'  Running rolling backtest on test set...')
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

    # Extract numpy arrays
    pred_np = forecast.values().flatten()
    actual_np = test_series.values().flatten()

    # Trim to same length (backtest may be slightly shorter)
    min_len = min(len(pred_np), len(actual_np))
    pred_np = pred_np[:min_len]
    actual_np = actual_np[:min_len]

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    nbeats_metrics = evaluate(actual_np, pred_np, f'N-BEATS (trained, {args.subsystem})')

    # ------------------------------------------------------------------
    # Load zero-shot results for comparison
    # ------------------------------------------------------------------
    all_metrics = {f'N-BEATS (trained)': nbeats_metrics}

    zs_path = os.path.join(OUTPUT_DIR, f'benchmark_{args.subsystem}_{args.horizon}h.csv')
    if os.path.exists(zs_path):
        zs_df = pd.read_csv(zs_path, index_col=0)
        for model_name in zs_df.index:
            all_metrics[f'{model_name} (zero-shot)'] = zs_df.loc[model_name].to_dict()

    ft_path = os.path.join(OUTPUT_DIR, f'finetune_{args.subsystem}_{args.horizon}h.csv')
    if os.path.exists(ft_path):
        ft_df = pd.read_csv(ft_path, index_col=0)
        if 'Chronos-2 (fine-tuned)' in ft_df.index:
            all_metrics['Chronos-2 (fine-tuned)'] = ft_df.loc['Chronos-2 (fine-tuned)'].to_dict()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f'\n{"="*70}')
    print(f'  FULL COMPARISON: {args.subsystem}, {args.horizon}h')
    print(f'{"="*70}')
    results_df = pd.DataFrame(all_metrics).T.round(2)
    if 'MAPE' in results_df.columns:
        results_df = results_df.sort_values('MAPE')
    print(results_df.to_string())

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR,
                            f'nbeats_comparison_{args.subsystem}_{args.horizon}h.csv')
    results_df.to_csv(csv_path)
    print(f'\nCSV: {csv_path}')

    return nbeats_metrics


if __name__ == '__main__':
    main()
