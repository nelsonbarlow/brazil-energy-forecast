#!/usr/bin/env python3
"""
Probabilistic evaluation of foundation model forecasts.

Computes CRPS, calibration, and prediction interval metrics from
quantile outputs that the models already produce.

Usage:
    python scripts/probabilistic_eval.py                    # SE, 24h
    python scripts/probabilistic_eval.py --subsystem NE     # Nordeste
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
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results')

# ---------------------------------------------------------------------------
# CRPS and calibration from quantile forecasts
# ---------------------------------------------------------------------------
def pinball_loss(y, q, tau):
    """Quantile (pinball) loss."""
    delta = y - q
    return np.where(delta >= 0, tau * delta, (tau - 1) * delta)


def crps_quantile(y, quantiles, tau_levels):
    """
    CRPS approximated from quantile forecasts.
    y: actual values, shape (n,)
    quantiles: predicted quantiles, shape (n, K)
    tau_levels: quantile levels, shape (K,)
    """
    n, K = quantiles.shape
    crps = np.zeros(n)
    for k in range(K):
        crps += pinball_loss(y, quantiles[:, k], tau_levels[k])
    return crps * (2.0 / K)


def calibration(y, quantiles, tau_levels):
    """
    Empirical coverage at each quantile level.
    Perfect calibration: empirical coverage = tau_level.
    """
    results = {}
    for k, tau in enumerate(tau_levels):
        coverage = np.mean(y <= quantiles[:, k])
        results[f'{tau:.1f}'] = coverage
    return results


def prediction_interval_width(quantiles, lower_idx, upper_idx):
    """Average width of the prediction interval."""
    return np.mean(quantiles[:, upper_idx] - quantiles[:, lower_idx])


def winkler_score(y, lower, upper, alpha):
    """Winkler score for prediction intervals."""
    width = upper - lower
    penalty_lower = (2.0 / alpha) * (lower - y) * (y < lower)
    penalty_upper = (2.0 / alpha) * (y - upper) * (y > upper)
    return np.mean(width + penalty_lower + penalty_upper)

# ---------------------------------------------------------------------------
# Model runners — return full quantile forecasts
# ---------------------------------------------------------------------------
def run_chronos_quantiles(full_load, test_start_idx, test_len, ctx_len, horizon, device):
    from chronos import BaseChronosPipeline
    model = BaseChronosPipeline.from_pretrained(
        'amazon/chronos-2', device_map=device, dtype=torch.float32,
    )
    all_quantiles = []  # list of (horizon, n_quantiles) arrays
    all_actuals = []

    for i in tqdm(range(0, test_len, horizon), desc='Chronos-2 (quantiles)'):
        ctx = torch.tensor(
            full_load[test_start_idx + i - ctx_len : test_start_idx + i],
            dtype=torch.float32,
        ).reshape(1, 1, -1)
        actual_horizon = min(horizon, test_len - i)
        forecasts = model.predict(ctx, prediction_length=actual_horizon)
        # shape: (n_variates, n_quantiles, pred_len)
        f = forecasts[0]  # (1, n_quantiles, actual_horizon)
        q = f[0, :, :].numpy().T  # (actual_horizon, n_quantiles)
        actuals = full_load[test_start_idx + i : test_start_idx + i + actual_horizon]
        all_quantiles.append(q)
        all_actuals.append(actuals)

    quantiles = np.concatenate(all_quantiles, axis=0)[:test_len]
    actuals = np.concatenate(all_actuals)[:test_len]
    # Chronos-2 quantile levels: evenly spaced, infer from count
    n_q = quantiles.shape[1]
    tau_levels = np.linspace(1/(n_q+1), n_q/(n_q+1), n_q)
    return actuals, quantiles, tau_levels


def run_tirex_quantiles(full_load, test_start_idx, test_len, ctx_len, horizon, device):
    from tirex import load_model
    model = load_model('NX-AI/TiRex')
    all_quantiles = []
    all_actuals = []

    for i in tqdm(range(0, test_len, horizon), desc='TiRex (quantiles)'):
        ctx = torch.tensor(
            full_load[test_start_idx + i - ctx_len : test_start_idx + i],
            dtype=torch.float32,
        ).unsqueeze(0)
        actual_horizon = min(horizon, test_len - i)
        quantiles_out, mean = model.forecast(context=ctx, prediction_length=actual_horizon)
        # quantiles_out shape: (1, n_quantiles, pred_len)
        q = quantiles_out[0, :, :actual_horizon].numpy().T  # (actual_horizon, n_quantiles)
        actuals = full_load[test_start_idx + i : test_start_idx + i + actual_horizon]
        all_quantiles.append(q)
        all_actuals.append(actuals)

    quantiles = np.concatenate(all_quantiles, axis=0)[:test_len]
    actuals = np.concatenate(all_actuals)[:test_len]
    n_q = quantiles.shape[1]
    tau_levels = np.linspace(1/(n_q+1), n_q/(n_q+1), n_q)
    return actuals, quantiles, tau_levels


def run_moirai_quantiles(full_load, test_start_idx, test_len, ctx_len, horizon, device):
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained('Salesforce/moirai-2.0-R-small'),
        prediction_length=horizon,
        context_length=ctx_len,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    # Moirai 2.0 quantile levels
    tau_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    all_quantiles = []
    all_actuals = []

    for i in tqdm(range(0, test_len, horizon), desc='Moirai 2.0 (quantiles)'):
        ctx_vals = full_load[test_start_idx + i - ctx_len : test_start_idx + i]
        actual_horizon = min(horizon, test_len - i)
        forecast = model.predict([ctx_vals.astype(np.float32)])
        # shape: (1, 9, pred_len)
        q = forecast[0, :, :actual_horizon].T  # (actual_horizon, 9)
        actuals = full_load[test_start_idx + i : test_start_idx + i + actual_horizon]
        all_quantiles.append(q)
        all_actuals.append(actuals)

    quantiles = np.concatenate(all_quantiles, axis=0)[:test_len]
    actuals = np.concatenate(all_actuals)[:test_len]
    return actuals, quantiles, tau_levels

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
    parser = argparse.ArgumentParser(description='Probabilistic evaluation of forecasts')
    parser.add_argument('--subsystem', type=str, default='SE',
                        choices=['SE', 'S', 'NE', 'N'])
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--context-length', type=int, default=720)
    parser.add_argument('--test-days', type=int, default=365)
    parser.add_argument('--models', nargs='+',
                        choices=['chronos', 'tirex', 'moirai'],
                        default=['chronos', 'tirex', 'moirai'])
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

    print(f'Subsystem: {args.subsystem}, Horizon: {args.horizon}h, Device: {device}')

    # Load data
    df = load_data(args.subsystem, args.test_days)
    full_load = df['load_mw'].values.astype(np.float32)
    test_start_idx = len(full_load) - args.test_days * 24
    test_len = args.test_days * 24

    runners = {
        'chronos': ('Chronos-2', run_chronos_quantiles),
        'tirex':   ('TiRex', run_tirex_quantiles),
        'moirai':  ('Moirai 2.0', run_moirai_quantiles),
    }

    all_results = {}

    for model_key in args.models:
        name, runner = runners[model_key]
        print(f'\n{"━"*60}')
        print(f'  {name}')
        print(f'{"━"*60}')
        try:
            actuals, quantiles, tau_levels = runner(
                full_load, test_start_idx, test_len,
                args.context_length, args.horizon, device,
            )
            n_q = len(tau_levels)
            print(f'  Quantile levels ({n_q}): {tau_levels.round(3)}')

            # CRPS
            crps_vals = crps_quantile(actuals, quantiles, tau_levels)
            mean_crps = crps_vals.mean()

            # Calibration
            cal = calibration(actuals, quantiles, tau_levels)

            # Find closest indices to 10th and 90th percentiles for 80% PI
            idx_10 = np.argmin(np.abs(tau_levels - 0.1))
            idx_90 = np.argmin(np.abs(tau_levels - 0.9))
            pi_width = prediction_interval_width(quantiles, idx_10, idx_90)
            pi_coverage = np.mean(
                (actuals >= quantiles[:, idx_10]) & (actuals <= quantiles[:, idx_90])
            )
            ws = winkler_score(
                actuals, quantiles[:, idx_10], quantiles[:, idx_90], alpha=0.2
            )

            # Median index for point forecast comparison
            idx_50 = np.argmin(np.abs(tau_levels - 0.5))
            median_preds = quantiles[:, idx_50]
            from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
            mae = mean_absolute_error(actuals, median_preds)
            mape = mean_absolute_percentage_error(actuals, median_preds) * 100

            results = {
                'CRPS (MW)': mean_crps,
                'MAE (MW)': mae,
                'MAPE': mape,
                '80% PI Coverage': pi_coverage * 100,
                '80% PI Width (MW)': pi_width,
                'Winkler Score': ws,
            }
            all_results[name] = results

            print(f'\n  Results:')
            print(f'    CRPS:              {mean_crps:,.0f} MW')
            print(f'    MAE (median):      {mae:,.0f} MW')
            print(f'    MAPE (median):     {mape:.2f}%')
            print(f'    80% PI Coverage:   {pi_coverage*100:.1f}% (ideal: 80%)')
            print(f'    80% PI Width:      {pi_width:,.0f} MW')
            print(f'    Winkler Score:     {ws:,.0f}')

            # Calibration table
            print(f'\n  Calibration (ideal: level = coverage):')
            for level, cov in cal.items():
                ideal = float(level)
                delta = cov - ideal
                marker = '  ' if abs(delta) < 0.05 else ' *'
                print(f'    τ={level}: {cov:.3f} (Δ={delta:+.3f}){marker}')

        except Exception as e:
            print(f'  FAILED: {e}')
            import traceback
            traceback.print_exc()

    if not all_results:
        print('\nNo models ran.')
        sys.exit(1)

    # Summary table
    print(f'\n{"="*70}')
    print(f'  PROBABILISTIC EVALUATION: {args.subsystem}, {args.horizon}h')
    print(f'{"="*70}')
    results_df = pd.DataFrame(all_results).T.round(2)
    print(results_df.to_string())

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR,
                            f'probabilistic_{args.subsystem}_{args.horizon}h.csv')
    results_df.to_csv(csv_path)
    print(f'\nCSV: {csv_path}')


if __name__ == '__main__':
    main()
