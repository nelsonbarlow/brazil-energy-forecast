#!/usr/bin/env python3
"""
Chronos model scaling benchmark: bolt-tiny (9M) → bolt-mini (21M) →
bolt-small (48M) → chronos-2 (120M) on SE subsystem.

Usage:
    python scripts/chronos_scaling.py
    python scripts/chronos_scaling.py --subsystem S
"""

import argparse
import glob
import os
import sys
import json
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

CHRONOS_MODELS = [
    ('Chronos-Bolt-Tiny (9M)', 'amazon/chronos-bolt-tiny', 9),
    ('Chronos-Bolt-Mini (21M)', 'amazon/chronos-bolt-mini', 21),
    ('Chronos-Bolt-Small (48M)', 'amazon/chronos-bolt-small', 48),
    ('Chronos-2 (120M)', 'amazon/chronos-2', 120),
]

HORIZON = 24
CONTEXT_LENGTH = 720


def _naive_forecast(actual, seasonality=1):
    return actual[:-seasonality]


def evaluate(actual, predicted, model_name):
    actual = np.asarray(actual).squeeze()
    predicted = np.asarray(predicted).squeeze()
    naive_err = mean_absolute_error(actual[24:], _naive_forecast(actual, 24))
    metrics = {
        'MAE (MW)': mean_absolute_error(actual, predicted),
        'RMSE (MW)': root_mean_squared_error(actual, predicted),
        'MAPE': mean_absolute_percentage_error(actual, predicted) * 100,
        'MASE': mean_absolute_error(actual, predicted) / naive_err,
        'RMSSE': np.sqrt(mean_squared_error(actual, predicted) /
                         mean_squared_error(actual[24:], _naive_forecast(actual, 24))),
        'R2': r2_score(actual, predicted),
    }
    pct = {'MAPE'}
    print(f'\n{"─"*60}')
    print(f'  {model_name}')
    print(f'{"─"*60}')
    for k, v in metrics.items():
        print(f'  {k:>12s}: {v:.2f}{"%" if k in pct else ""}')
    return metrics


def load_data(subsystem):
    pattern = os.path.join(PROCESSED_DIR, f'ons_hourly_load_{subsystem.lower()}_*.csv')
    files = glob.glob(pattern)
    if not files:
        print(f'No data. Run: python scripts/download_ons.py --subsystem {subsystem}')
        sys.exit(1)
    df = pd.read_csv(files[0])
    df['datetime'] = pd.to_datetime(df['datetime'])
    if 'subsystem' in df.columns and len(df['subsystem'].unique()) > 1:
        df = df[df['subsystem'] == subsystem.upper()].reset_index(drop=True)
    return df.sort_values('datetime').reset_index(drop=True)


def run_chronos_model(model_id, full_load, test_start_idx, test_len, device):
    from chronos import BaseChronosPipeline
    model = BaseChronosPipeline.from_pretrained(
        model_id, device_map=device, dtype=torch.float32,
    )
    # Detect if this is a Chronos-2 (multivariate, 3D) or Bolt (2D) model
    is_chronos2 = 'chronos-2' in model_id
    preds = []
    desc = model_id.split('/')[-1]
    for i in tqdm(range(0, test_len, HORIZON), desc=desc):
        ctx_np = full_load[test_start_idx + i - CONTEXT_LENGTH : test_start_idx + i]
        ctx = torch.tensor(ctx_np, dtype=torch.float32)
        if is_chronos2:
            ctx = ctx.reshape(1, 1, -1)  # (batch, variates, seq_len)
        else:
            ctx = ctx.unsqueeze(0)  # (batch, seq_len)
        actual_horizon = min(HORIZON, test_len - i)
        forecasts = model.predict(ctx, prediction_length=actual_horizon)
        f = forecasts[0]
        median_idx = f.shape[-2] // 2
        if is_chronos2:
            preds.extend(f[0, median_idx, :actual_horizon].tolist())
        else:
            preds.extend(f[median_idx, :actual_horizon].tolist())
    return np.array(preds[:test_len])


def main():
    parser = argparse.ArgumentParser(description='Chronos scaling benchmark')
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

    df = load_data(args.subsystem)
    full_load = df['load_mw'].values
    test_hours = 365 * 24
    test_start_idx = len(full_load) - test_hours
    test_actuals = full_load[test_start_idx:]
    test_len = len(test_actuals)

    print(f'Subsystem: {args.subsystem}, Device: {device}')
    print(f'Test: {test_len} hours, Context: {CONTEXT_LENGTH}h')

    all_metrics = {}
    for display_name, model_id, params_m in CHRONOS_MODELS:
        print(f'\n{"━"*60}')
        print(f'  {display_name} ({model_id})')
        print(f'{"━"*60}')
        try:
            preds = run_chronos_model(model_id, full_load, test_start_idx,
                                      test_len, device)
            metrics = evaluate(test_actuals, preds, display_name)
            metrics['Params (M)'] = params_m
            all_metrics[display_name] = metrics
        except Exception as e:
            print(f'  FAILED: {e}')
            import traceback
            traceback.print_exc()

    # Summary
    print(f'\n{"="*70}')
    print(f'  CHRONOS SCALING: {args.subsystem}, {HORIZON}h')
    print(f'{"="*70}')
    results_df = pd.DataFrame(all_metrics).T.round(2)
    print(results_df.to_string())

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR,
                            f'chronos_scaling_{args.subsystem}_{HORIZON}h.csv')
    results_df.to_csv(csv_path)
    print(f'\nCSV: {csv_path}')

    # Also save as JSON for easy consumption
    json_path = os.path.join(OUTPUT_DIR,
                             f'chronos_scaling_{args.subsystem}_{HORIZON}h.json')
    with open(json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f'JSON: {json_path}')

    # Plot scaling curve
    if len(all_metrics) > 1:
        names = list(all_metrics.keys())
        params = [all_metrics[n]['Params (M)'] for n in names]
        mapes = [all_metrics[n]['MAPE'] for n in names]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(params, mapes, 'bo-', markersize=8, linewidth=2)
        for p, m, n in zip(params, mapes, names):
            short = n.split('(')[0].strip()
            ax.annotate(f'{short}\n{m:.2f}%', (p, m),
                        textcoords='offset points', xytext=(0, 12),
                        ha='center', fontsize=9)
        ax.set_xlabel('Parameters (M)')
        ax.set_ylabel('MAPE (%)')
        ax.set_title(f'Chronos Model Scaling — {args.subsystem} Subsystem, {HORIZON}h')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        png_path = os.path.join(OUTPUT_DIR,
                                f'chronos_scaling_{args.subsystem}_{HORIZON}h.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f'Plot: {png_path}')


if __name__ == '__main__':
    main()
