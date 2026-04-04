#!/usr/bin/env python3
"""
Context length ablation: how much history does Chronos-2 need?

Tests context windows from 24h (1 day) to 2160h (90 days) and plots
the accuracy curve.

Usage:
    python scripts/context_ablation.py                  # SE subsystem
    python scripts/context_ablation.py --subsystem NE   # other subsystem
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
    mean_absolute_error, mean_absolute_percentage_error, r2_score,
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results')

HORIZON = 24

# Context lengths to test (hours)
CONTEXT_LENGTHS = [24, 72, 168, 336, 720, 1440, 2160]
CONTEXT_LABELS = ['1d', '3d', '1w', '2w', '30d', '60d', '90d']

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(subsystem, test_days=365):
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
    df = df.sort_values('datetime').reset_index(drop=True)
    test_cutoff = df['datetime'].max() - pd.Timedelta(days=test_days)
    train_df = df[df['datetime'] <= test_cutoff]
    return df, len(train_df)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Context length ablation')
    parser.add_argument('--subsystem', type=str, default='SE',
                        choices=['SE', 'S', 'NE', 'N'])
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

    print(f'Subsystem: {args.subsystem}, Device: {device}')
    print(f'Context lengths: {CONTEXT_LENGTHS}')

    # Load data
    df, test_start_idx = load_data(args.subsystem)
    full_load = df['load_mw'].values.astype(np.float32)
    test_actuals = full_load[test_start_idx:]
    test_len = len(test_actuals)

    # Load model once
    from chronos import BaseChronosPipeline
    print('\nLoading Chronos-2...')
    model = BaseChronosPipeline.from_pretrained(
        'amazon/chronos-2', device_map=device, dtype=torch.float32,
    )

    # Run for each context length
    results = {}
    for ctx_len, label in zip(CONTEXT_LENGTHS, CONTEXT_LABELS):
        # Skip if not enough history
        if test_start_idx < ctx_len:
            print(f'\n  {label} ({ctx_len}h): skipped — not enough history')
            continue

        print(f'\n  {label} ({ctx_len}h)...')
        preds = []
        for i in tqdm(range(0, test_len, HORIZON), desc=f'ctx={label}'):
            ctx = torch.tensor(
                full_load[test_start_idx + i - ctx_len : test_start_idx + i],
                dtype=torch.float32,
            ).reshape(1, 1, -1)
            actual_h = min(HORIZON, test_len - i)
            forecasts = model.predict(ctx, prediction_length=actual_h)
            f = forecasts[0]
            median_idx = f.shape[1] // 2
            preds.extend(f[0, median_idx, :actual_h].tolist())

        preds = np.array(preds[:test_len])
        mape = mean_absolute_percentage_error(test_actuals, preds) * 100
        mae = mean_absolute_error(test_actuals, preds)
        r2 = r2_score(test_actuals, preds)

        results[label] = {
            'Context (h)': ctx_len,
            'MAPE': mape,
            'MAE (MW)': mae,
            'R2': r2,
        }
        print(f'    MAPE: {mape:.2f}%, MAE: {mae:.0f} MW, R²: {r2:.3f}')

    # Summary table
    print(f'\n{"="*60}')
    print(f'  CONTEXT LENGTH ABLATION: {args.subsystem}, {HORIZON}h horizon')
    print(f'{"="*60}')
    results_df = pd.DataFrame(results).T.round(3)
    print(results_df.to_string())

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f'context_ablation_{args.subsystem}.csv')
    results_df.to_csv(csv_path)
    print(f'\nCSV: {csv_path}')

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ctx_hours = [results[l]['Context (h)'] for l in results]
    mapes = [results[l]['MAPE'] for l in results]

    ax1.plot(ctx_hours, mapes, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Context Length (hours)', fontsize=12)
    ax1.set_ylabel('MAPE (%)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xscale('log')
    ax1.set_xticks(ctx_hours)
    ax1.set_xticklabels([f'{h}h\n({l})' for h, l in zip(ctx_hours, results.keys())],
                         fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Annotate each point
    for h, m in zip(ctx_hours, mapes):
        ax1.annotate(f'{m:.2f}%', (h, m), textcoords='offset points',
                     xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')

    plt.title(f'Chronos-2 Context Length Ablation — {args.subsystem} Subsystem',
              fontsize=13, fontweight='bold')
    plt.tight_layout()

    chart_path = os.path.join(OUTPUT_DIR, f'context_ablation_{args.subsystem}.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f'Chart: {chart_path}')


if __name__ == '__main__':
    main()
