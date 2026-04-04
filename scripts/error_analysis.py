#!/usr/bin/env python3
"""
Error analysis: break down Chronos-2 forecast errors by hour-of-day,
day-of-week, and identify worst prediction days.

Usage:
    python scripts/error_analysis.py                    # SE, uses saved predictions
    python scripts/error_analysis.py --subsystem NE     # other subsystem (re-runs model)
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
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results')

CONTEXT_LENGTH = 720
HORIZON = 24

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
    test_df = df[df['datetime'] > test_cutoff].reset_index(drop=True)
    return df, train_df, test_df

# ---------------------------------------------------------------------------
# Generate predictions
# ---------------------------------------------------------------------------
def get_predictions(full_load, test_start_idx, test_len, device):
    from chronos import BaseChronosPipeline
    model = BaseChronosPipeline.from_pretrained(
        'amazon/chronos-2', device_map=device, dtype=torch.float32,
    )
    preds = []
    for i in tqdm(range(0, test_len, HORIZON), desc='Chronos-2'):
        ctx = torch.tensor(
            full_load[test_start_idx + i - CONTEXT_LENGTH : test_start_idx + i],
            dtype=torch.float32,
        ).reshape(1, 1, -1)
        actual_h = min(HORIZON, test_len - i)
        forecasts = model.predict(ctx, prediction_length=actual_h)
        f = forecasts[0]
        median_idx = f.shape[1] // 2
        preds.extend(f[0, median_idx, :actual_h].tolist())
    return np.array(preds[:test_len])

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Error analysis by hour/day/event')
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

    # Load data
    df, train_df, test_df = load_data(args.subsystem)
    full_load = df['load_mw'].values.astype(np.float32)
    test_start_idx = len(train_df)
    test_actuals = full_load[test_start_idx:]

    # Generate predictions
    print(f'Generating Chronos-2 predictions for {args.subsystem}...')
    preds = get_predictions(full_load, test_start_idx, len(test_actuals), device)

    # Build analysis DataFrame
    analysis = test_df[['datetime', 'load_mw']].copy()
    analysis = analysis.iloc[:len(preds)].reset_index(drop=True)
    analysis['predicted'] = preds
    analysis['error'] = analysis['predicted'] - analysis['load_mw']
    analysis['abs_error'] = np.abs(analysis['error'])
    analysis['pct_error'] = np.abs(analysis['error']) / analysis['load_mw'] * 100
    analysis['hour'] = analysis['datetime'].dt.hour
    analysis['dayofweek'] = analysis['datetime'].dt.dayofweek  # 0=Mon, 6=Sun
    analysis['day_name'] = analysis['datetime'].dt.day_name()
    analysis['date'] = analysis['datetime'].dt.date
    analysis['is_weekend'] = analysis['dayofweek'].isin([5, 6])

    # ------------------------------------------------------------------
    # 1. MAPE by hour of day
    # ------------------------------------------------------------------
    hourly = analysis.groupby('hour').agg(
        MAPE=('pct_error', 'mean'),
        MAE=('abs_error', 'mean'),
        count=('pct_error', 'count'),
    ).round(2)

    print(f'\n{"="*50}')
    print(f'  MAPE by Hour of Day ({args.subsystem})')
    print(f'{"="*50}')
    print(hourly[['MAPE', 'MAE']].to_string())

    # ------------------------------------------------------------------
    # 2. MAPE by day of week
    # ------------------------------------------------------------------
    daily = analysis.groupby('day_name').agg(
        MAPE=('pct_error', 'mean'),
        MAE=('abs_error', 'mean'),
        count=('pct_error', 'count'),
    )
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily = daily.reindex(day_order).round(2)

    print(f'\n{"="*50}')
    print(f'  MAPE by Day of Week ({args.subsystem})')
    print(f'{"="*50}')
    print(daily[['MAPE', 'MAE']].to_string())

    # ------------------------------------------------------------------
    # 3. Weekend vs weekday
    # ------------------------------------------------------------------
    weekday_mape = analysis[~analysis['is_weekend']]['pct_error'].mean()
    weekend_mape = analysis[analysis['is_weekend']]['pct_error'].mean()

    print(f'\n{"="*50}')
    print(f'  Weekday vs Weekend ({args.subsystem})')
    print(f'{"="*50}')
    print(f'  Weekday MAPE: {weekday_mape:.2f}%')
    print(f'  Weekend MAPE: {weekend_mape:.2f}%')

    # ------------------------------------------------------------------
    # 4. Worst 10 days
    # ------------------------------------------------------------------
    daily_mape = analysis.groupby('date').agg(
        MAPE=('pct_error', 'mean'),
        MAE=('abs_error', 'mean'),
        mean_load=('load_mw', 'mean'),
    ).round(2)
    worst = daily_mape.nlargest(10, 'MAPE')

    print(f'\n{"="*50}')
    print(f'  10 Worst Prediction Days ({args.subsystem})')
    print(f'{"="*50}')
    for date, row in worst.iterrows():
        dow = pd.Timestamp(date).day_name()
        print(f'  {date} ({dow:>9s}): MAPE={row["MAPE"]:.2f}%, '
              f'MAE={row["MAE"]:,.0f} MW, Load={row["mean_load"]:,.0f} MW')

    # ------------------------------------------------------------------
    # 5. Save plots
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Hour of day
    axes[0, 0].bar(hourly.index, hourly['MAPE'], color='steelblue')
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('MAPE (%)')
    axes[0, 0].set_title('MAPE by Hour of Day')
    axes[0, 0].set_xticks(range(0, 24, 2))
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Day of week
    colors = ['steelblue'] * 5 + ['coral'] * 2
    axes[0, 1].bar(range(7), daily['MAPE'].values, color=colors)
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[0, 1].set_ylabel('MAPE (%)')
    axes[0, 1].set_title('MAPE by Day of Week (orange = weekend)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Error distribution
    axes[1, 0].hist(analysis['pct_error'], bins=80, color='steelblue', alpha=0.7)
    axes[1, 0].axvline(x=analysis['pct_error'].median(), color='red',
                        linestyle='--', label=f'Median: {analysis["pct_error"].median():.2f}%')
    axes[1, 0].set_xlabel('Absolute Percentage Error (%)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Daily MAPE over test year
    daily_mape_sorted = daily_mape.sort_index()
    axes[1, 1].plot(daily_mape_sorted.index, daily_mape_sorted['MAPE'],
                     color='steelblue', linewidth=0.8)
    axes[1, 1].axhline(y=analysis['pct_error'].mean(), color='red',
                         linestyle='--', alpha=0.5, label=f'Mean: {analysis["pct_error"].mean():.2f}%')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Daily MAPE (%)')
    axes[1, 1].set_title('Daily MAPE Over Test Year')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.suptitle(f'Chronos-2 Error Analysis — {args.subsystem} Subsystem, 24h Horizon',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    chart_path = os.path.join(OUTPUT_DIR, f'error_analysis_{args.subsystem}.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f'\nChart: {chart_path}')

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, f'error_analysis_{args.subsystem}.csv')
    summary = pd.DataFrame({
        'Metric': ['Overall MAPE', 'Weekday MAPE', 'Weekend MAPE',
                   'Best Hour MAPE', 'Worst Hour MAPE',
                   'Best Day MAPE', 'Worst Day MAPE'],
        'Value': [
            f'{analysis["pct_error"].mean():.2f}%',
            f'{weekday_mape:.2f}%',
            f'{weekend_mape:.2f}%',
            f'{hourly["MAPE"].min():.2f}% (hour {hourly["MAPE"].idxmin()})',
            f'{hourly["MAPE"].max():.2f}% (hour {hourly["MAPE"].idxmax()})',
            f'{daily["MAPE"].min():.2f}% ({daily["MAPE"].idxmin()})',
            f'{daily["MAPE"].max():.2f}% ({daily["MAPE"].idxmax()})',
        ]
    })
    summary.to_csv(csv_path, index=False)
    print(f'CSV: {csv_path}')


if __name__ == '__main__':
    main()
