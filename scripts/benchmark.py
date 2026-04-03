#!/usr/bin/env python3
"""
Benchmark foundation models on Brazilian electricity load forecasting.

Uses ONS hourly load data. Predicts the next 24 hours of load for a
given subsystem using a rolling context window.

Usage:
    python scripts/benchmark.py                              # run all models, SE subsystem
    python scripts/benchmark.py --subsystem NE               # Nordeste
    python scripts/benchmark.py --models chronos tirex       # specific models
    python scripts/benchmark.py --horizon 1                  # 1-hour ahead only
    python scripts/benchmark.py --context-length 720         # 30 days of hourly context
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

AVAILABLE_MODELS = ['chronos', 'tirex', 'moirai', 'naive']

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _naive_forecast(actual, seasonality=1):
    return actual[:-seasonality]

def rmsse(actual, predicted, seasonality=24):
    """RMSSE with daily seasonality (24h)."""
    q = mean_squared_error(actual, predicted) / mean_squared_error(
        actual[seasonality:], _naive_forecast(actual, seasonality))
    return np.sqrt(q)

def mase(actual, predicted, seasonality=24):
    """MASE with daily seasonality (24h)."""
    return mean_absolute_error(actual, predicted) / mean_absolute_error(
        actual[seasonality:], _naive_forecast(actual, seasonality))

def evaluate(actual, predicted, model_name):
    actual = np.asarray(actual).squeeze()
    predicted = np.asarray(predicted).squeeze()

    metrics = {
        'MAE (MW)':  mean_absolute_error(actual, predicted),
        'RMSE (MW)': root_mean_squared_error(actual, predicted),
        'MAPE':      mean_absolute_percentage_error(actual, predicted) * 100,
        'MASE':      mase(actual, predicted),
        'RMSSE':     rmsse(actual, predicted),
        'R2':        r2_score(actual, predicted),
    }

    print(f'\n{"─"*60}')
    print(f'  {model_name}')
    print(f'{"─"*60}')
    for k, v in metrics.items():
        fmt = f'{v:.2f}%' if k == 'MAPE' else f'{v:.2f}'
        print(f'  {k:>12s}: {fmt}')
    return metrics

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(subsystem, test_days=365):
    """Load processed ONS data for a subsystem, split into train/test."""
    pattern = os.path.join(PROCESSED_DIR, f'ons_hourly_load_{subsystem.lower()}_*.csv')
    files = glob.glob(pattern)
    if not files:
        # Try the 'all' file
        pattern = os.path.join(PROCESSED_DIR, 'ons_hourly_load_all_*.csv')
        files = glob.glob(pattern)

    if not files:
        print(f'No processed data found. Run: python scripts/download_ons.py --subsystem {subsystem}')
        sys.exit(1)

    df = pd.read_csv(files[0])
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Filter to subsystem if using the 'all' file
    if 'subsystem' in df.columns and len(df['subsystem'].unique()) > 1:
        df = df[df['subsystem'] == subsystem.upper()].reset_index(drop=True)

    df = df.sort_values('datetime').reset_index(drop=True)

    # Split: last test_days days as test
    test_cutoff = df['datetime'].max() - pd.Timedelta(days=test_days)
    train_df = df[df['datetime'] <= test_cutoff]
    test_df = df[df['datetime'] > test_cutoff]

    return df, train_df, test_df

# ---------------------------------------------------------------------------
# Model runners — predict next `horizon` hours given context
# ---------------------------------------------------------------------------
def run_naive(full_load, test_start_idx, test_len, ctx_len, horizon):
    """Naive baseline: predict same hour from 7 days ago (weekly seasonality)."""
    preds = []
    for i in range(0, test_len, horizon):
        actual_horizon = min(horizon, test_len - i)
        pred = []
        for h in range(actual_horizon):
            # Same hour, 7 days ago
            idx = test_start_idx + i + h - 168  # 7 * 24
            pred.append(full_load[idx] if idx >= 0 else full_load[test_start_idx])
        preds.extend(pred)
    return np.array(preds[:test_len])


def run_chronos(full_load, test_start_idx, test_len, ctx_len, horizon, device):
    from chronos import BaseChronosPipeline
    model = BaseChronosPipeline.from_pretrained(
        'amazon/chronos-2', device_map=device, dtype=torch.float32,
    )
    preds = []
    for i in tqdm(range(0, test_len, horizon), desc='Chronos-2'):
        ctx = torch.tensor(
            full_load[test_start_idx + i - ctx_len : test_start_idx + i],
            dtype=torch.float32,
        ).reshape(1, 1, -1)
        actual_horizon = min(horizon, test_len - i)
        forecasts = model.predict(ctx, prediction_length=actual_horizon)
        f = forecasts[0]
        median_idx = f.shape[1] // 2
        preds.extend(f[0, median_idx, :actual_horizon].tolist())
    return np.array(preds[:test_len])


def run_tirex(full_load, test_start_idx, test_len, ctx_len, horizon, device):
    from tirex import load_model
    model = load_model('NX-AI/TiRex')
    preds = []
    for i in tqdm(range(0, test_len, horizon), desc='TiRex'):
        ctx = torch.tensor(
            full_load[test_start_idx + i - ctx_len : test_start_idx + i],
            dtype=torch.float32,
        ).unsqueeze(0)
        actual_horizon = min(horizon, test_len - i)
        quantiles, mean = model.forecast(context=ctx, prediction_length=actual_horizon)
        preds.extend(mean.squeeze().tolist() if actual_horizon > 1
                     else [mean.squeeze().item()])
    return np.array(preds[:test_len])


def run_moirai(full_load, test_start_idx, test_len, ctx_len, horizon, device):
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained('Salesforce/moirai-2.0-R-small'),
        prediction_length=horizon,
        context_length=ctx_len,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    preds = []
    for i in tqdm(range(0, test_len, horizon), desc='Moirai 2.0'):
        ctx_vals = full_load[test_start_idx + i - ctx_len : test_start_idx + i]
        actual_horizon = min(horizon, test_len - i)
        forecast = model.predict([ctx_vals.astype(np.float32)])
        # Index 4 = median (0.5 quantile)
        preds.extend(forecast[0, 4, :actual_horizon].tolist())
    return np.array(preds[:test_len])


MODEL_RUNNERS = {
    'naive':   ('Naive (7d ago)',  None),
    'chronos': ('Chronos-2',      run_chronos),
    'tirex':   ('TiRex',          run_tirex),
    'moirai':  ('Moirai 2.0',     run_moirai),
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_chart(results, test_dates, test_actuals, output_path, subsystem):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Top: first 7 days of predictions vs actual
    n_plot = min(168, len(test_actuals))  # 7 days
    axes[0].plot(test_dates[:n_plot], test_actuals[:n_plot],
                 'k-', linewidth=1.5, label='Actual')
    colors = ['#FF5722', '#2196F3', '#4CAF50', '#9C27B0']
    for (name, preds), color in zip(results.items(), colors):
        if isinstance(preds, np.ndarray):
            axes[0].plot(test_dates[:n_plot], preds[:n_plot],
                         '--', color=color, linewidth=1, alpha=0.8, label=name)
    axes[0].set_ylabel('Load (MW)')
    axes[0].set_title(f'{subsystem} Subsystem - First 7 Days of Test Set')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom: bar chart of MAE
    model_names = list(results.keys())
    maes = [mean_absolute_error(test_actuals, p) for p in results.values()
            if isinstance(p, np.ndarray)]
    axes[1].bar(model_names[:len(maes)], maes, color=colors[:len(maes)])
    axes[1].set_ylabel('MAE (MW)')
    axes[1].set_title('Mean Absolute Error by Model')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\nChart saved to {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Benchmark foundation models on ONS load data')
    parser.add_argument('--subsystem', type=str, default='SE',
                        choices=['SE', 'S', 'NE', 'N'],
                        help='ONS subsystem (default: SE = Sudeste)')
    parser.add_argument('--models', nargs='+', choices=AVAILABLE_MODELS,
                        default=['naive', 'chronos', 'tirex', 'moirai'],
                        help='Models to run')
    parser.add_argument('--horizon', type=int, default=24,
                        help='Forecast horizon in hours (default: 24)')
    parser.add_argument('--context-length', type=int, default=720,
                        help='Context window in hours (default: 720 = 30 days)')
    parser.add_argument('--test-days', type=int, default=365,
                        help='Number of days for test set (default: 365)')
    parser.add_argument('--device', type=str, default=None,
                        help='Force device (cuda/mps/cpu)')
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
    print(f'Models: {", ".join(args.models)}')

    # Load data
    df, train_df, test_df = load_data(args.subsystem, args.test_days)
    full_load = df['load_mw'].values
    test_start_idx = len(train_df)
    test_actuals = full_load[test_start_idx:]
    test_dates = df['datetime'].values[test_start_idx:]
    test_len = len(test_actuals)

    print(f'\nTotal rows: {len(df):,}')
    print(f'Train: {len(train_df):,} rows')
    print(f'Test:  {test_len:,} rows ({args.test_days} days)')
    print(f'Mean load: {test_actuals.mean():,.0f} MW')

    # Run models
    all_metrics = {}
    all_preds = {}

    for model_key in args.models:
        display_name, runner = MODEL_RUNNERS[model_key]
        print(f'\n{"━"*60}')
        print(f'  Running {display_name}...')
        print(f'{"━"*60}')
        try:
            if model_key == 'naive':
                preds = run_naive(full_load, test_start_idx, test_len,
                                  args.context_length, args.horizon)
            else:
                preds = runner(full_load, test_start_idx, test_len,
                               args.context_length, args.horizon, device)
            all_preds[display_name] = preds
            all_metrics[display_name] = evaluate(test_actuals, preds, display_name)
        except Exception as e:
            print(f'  FAILED: {e}')

    if not all_metrics:
        print('\nNo models ran.')
        sys.exit(1)

    # Summary table
    print(f'\n{"="*70}')
    print(f'  RESULTS: {args.subsystem} subsystem, {args.horizon}h horizon')
    print(f'{"="*70}')
    results_df = pd.DataFrame(all_metrics).T.round(2)
    print(results_df.to_string())

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f'benchmark_{args.subsystem}_{args.horizon}h.csv')
    results_df.to_csv(csv_path)
    print(f'\nCSV: {csv_path}')

    chart_path = os.path.join(OUTPUT_DIR, f'benchmark_{args.subsystem}_{args.horizon}h.png')
    save_chart(all_preds, test_dates, test_actuals, chart_path, args.subsystem)


if __name__ == '__main__':
    main()
