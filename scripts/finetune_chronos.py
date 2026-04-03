#!/usr/bin/env python3
"""
Fine-tune Chronos-2 on ONS load data and compare with zero-shot.

Usage:
    python scripts/finetune_chronos.py                                # defaults
    python scripts/finetune_chronos.py --num-steps 2000 --lr 1e-6    # tuning
    python scripts/finetune_chronos.py --subsystem NE                 # other region
    python scripts/finetune_chronos.py --device cpu                   # MPS workaround
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
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results')

# ---------------------------------------------------------------------------
# Metrics
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
# Rolling prediction
# ---------------------------------------------------------------------------
def rolling_predict(pipeline, full_load, test_start_idx, test_len,
                    ctx_len, horizon, desc):
    preds = []
    for i in tqdm(range(0, test_len, horizon), desc=desc):
        ctx = torch.tensor(
            full_load[test_start_idx + i - ctx_len : test_start_idx + i],
            dtype=torch.float32,
        ).reshape(1, 1, -1)
        actual_h = min(horizon, test_len - i)
        forecasts = pipeline.predict(ctx, prediction_length=actual_h)
        f = forecasts[0]
        median_idx = f.shape[1] // 2
        preds.extend(f[0, median_idx, :actual_h].tolist())
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
    parser = argparse.ArgumentParser(description='Fine-tune Chronos-2 on ONS load data')
    parser.add_argument('--subsystem', type=str, default='SE',
                        choices=['SE', 'S', 'NE', 'N'])
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--context-length', type=int, default=720)
    parser.add_argument('--test-days', type=int, default=365)
    parser.add_argument('--val-days', type=int, default=60)
    parser.add_argument('--num-steps', type=int, default=1000,
                        help='Fine-tuning steps (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--finetune-mode', type=str, default='full',
                        choices=['full', 'lora'])
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # Device — default to CPU for fine-tuning (MPS doesn't support fused AdamW)
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'  # MPS fails for fine-tuning, safe default

    print(f'Subsystem: {args.subsystem}')
    print(f'Device: {device}')
    print(f'Fine-tune: {args.finetune_mode}, {args.num_steps} steps, LR={args.lr}')

    # ------------------------------------------------------------------
    # Load and split data
    # ------------------------------------------------------------------
    df = load_data(args.subsystem)
    full_load = df['load_mw'].values.astype(np.float32)

    test_hours = args.test_days * 24
    val_hours = args.val_days * 24
    test_start = len(full_load) - test_hours
    val_start = test_start - val_hours

    train_load = full_load[:val_start]
    val_load = full_load[val_start:test_start]
    test_actuals = full_load[test_start:]

    print(f'\nTrain: {len(train_load):,} hours ({len(train_load)//24:,} days)')
    print(f'Val:   {len(val_load):,} hours ({args.val_days} days)')
    print(f'Test:  {len(test_actuals):,} hours ({args.test_days} days)')

    # ------------------------------------------------------------------
    # Load base model
    # ------------------------------------------------------------------
    from chronos import Chronos2Pipeline

    print(f'\nLoading amazon/chronos-2...')
    base = Chronos2Pipeline.from_pretrained(
        'amazon/chronos-2',
        device_map=device,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------
    # Zero-shot baseline
    # ------------------------------------------------------------------
    print('\n--- Zero-shot baseline ---')
    zs_preds = rolling_predict(
        base, full_load, test_start, len(test_actuals),
        args.context_length, args.horizon, 'Chronos-2 (zero-shot)',
    )
    zs_metrics = evaluate(test_actuals, zs_preds, 'Chronos-2 (zero-shot)')

    # ------------------------------------------------------------------
    # Fine-tune
    # ------------------------------------------------------------------
    print(f'\n{"━"*60}')
    print(f'  Fine-tuning ({args.num_steps} steps)...')
    print(f'{"━"*60}')

    ft_pipeline = base.fit(
        inputs=[train_load],
        validation_inputs=[val_load],
        prediction_length=args.horizon,
        context_length=args.context_length,
        learning_rate=args.lr,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        finetune_mode=args.finetune_mode,
        output_dir=os.path.join(OUTPUT_DIR, f'chronos2-ft-{args.subsystem}'),
    )

    # ------------------------------------------------------------------
    # Evaluate fine-tuned model
    # ------------------------------------------------------------------
    print('\n--- Fine-tuned ---')
    ft_preds = rolling_predict(
        ft_pipeline, full_load, test_start, len(test_actuals),
        args.context_length, args.horizon, 'Chronos-2 (fine-tuned)',
    )
    ft_metrics = evaluate(test_actuals, ft_preds, 'Chronos-2 (fine-tuned)')

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    print(f'\n{"="*60}')
    print(f'  COMPARISON: {args.subsystem}, {args.horizon}h horizon')
    print(f'{"="*60}')
    comp = pd.DataFrame({
        'Chronos-2 (zero-shot)': zs_metrics,
        'Chronos-2 (fine-tuned)': ft_metrics,
    }).T.round(2)
    print(comp.to_string())

    improvement = (zs_metrics['MAPE'] - ft_metrics['MAPE']) / zs_metrics['MAPE'] * 100
    print(f'\nMAPE change: {zs_metrics["MAPE"]:.2f}% -> {ft_metrics["MAPE"]:.2f}% '
          f'({"improved" if improvement > 0 else "degraded"} by {abs(improvement):.1f}%)')

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f'finetune_{args.subsystem}_{args.horizon}h.csv')
    comp.to_csv(csv_path)
    print(f'CSV: {csv_path}')


if __name__ == '__main__':
    main()
