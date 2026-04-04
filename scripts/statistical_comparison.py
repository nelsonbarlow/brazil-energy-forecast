#!/usr/bin/env python3
"""
Statistical comparison of Chronos-2 (zero-shot) vs N-BEATS (tuned).

Computes:
  1. Per-window (24h) MAPE for both models
  2. Diebold-Mariano test for equal predictive accuracy
  3. Bootstrap confidence intervals on MAPE
  4. Paired t-test on per-window absolute percentage errors

Usage:
    python scripts/statistical_comparison.py
    python scripts/statistical_comparison.py --subsystem NE
"""

import argparse
import glob
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results')

HORIZON = 24
CONTEXT_LENGTH = 720
N_BOOTSTRAP = 10_000
CONFIDENCE = 0.95


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
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def run_chronos_windows(full_load, test_start_idx, test_len, device):
    """Run Chronos-2, return per-window predictions (n_windows × horizon)."""
    from chronos import BaseChronosPipeline
    model = BaseChronosPipeline.from_pretrained(
        'amazon/chronos-2', device_map=device, dtype=torch.float32,
    )
    windows = []
    for i in tqdm(range(0, test_len, HORIZON), desc='Chronos-2'):
        actual_horizon = min(HORIZON, test_len - i)
        if actual_horizon < HORIZON:
            break  # skip incomplete last window
        ctx = torch.tensor(
            full_load[test_start_idx + i - CONTEXT_LENGTH : test_start_idx + i],
            dtype=torch.float32,
        ).reshape(1, 1, -1)
        forecasts = model.predict(ctx, prediction_length=HORIZON)
        f = forecasts[0]
        median_idx = f.shape[1] // 2
        windows.append(f[0, median_idx, :HORIZON].numpy())
    return np.array(windows)


def run_nbeats_windows(full_load, test_start_idx, test_len, subsystem,
                       input_length=168, lr=5e-4, seed=42):
    """Train and run N-BEATS, return per-window predictions."""
    from darts import TimeSeries, concatenate
    from darts.models import NBEATSModel
    from pytorch_lightning.callbacks import EarlyStopping

    # Build full series for Darts
    df = pd.DataFrame({
        'datetime': pd.date_range('2019-01-01', periods=len(full_load), freq='h'),
        'load_mw': full_load,
    })
    series = TimeSeries.from_dataframe(
        df, time_col='datetime', value_cols='load_mw', freq='h',
        fill_missing_dates=True,
    )

    test_hours = test_len
    val_hours = 60 * 24
    test_series = series[-test_hours:]
    pre_test = series[:-test_hours]
    val_series = pre_test[-val_hours:]
    train_series = pre_test[:-val_hours]

    early_stopper = EarlyStopping(
        monitor='val_loss', patience=10, min_delta=1e-5, mode='min',
    )
    model = NBEATSModel(
        input_chunk_length=input_length,
        output_chunk_length=HORIZON,
        num_stacks=30, num_blocks=1, num_layers=4, layer_widths=256,
        n_epochs=200, batch_size=256,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={'lr': lr},
        lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_kwargs={'patience': 5, 'factor': 0.5, 'min_lr': 1e-6},
        pl_trainer_kwargs={
            'callbacks': [early_stopper],
            'accelerator': 'cpu',
        },
        random_state=seed,
        force_reset=True,
        save_checkpoints=True,
    )
    print(f'\n  Training N-BEATS (seed={seed}, lr={lr}, input={input_length})...')
    model.fit(series=train_series, val_series=val_series, verbose=True)

    print('  Running rolling backtest...')
    backtest = model.historical_forecasts(
        series=series,
        start=test_series.start_time(),
        forecast_horizon=HORIZON,
        stride=HORIZON,
        retrain=False,
        last_points_only=False,
        verbose=True,
    )

    # Extract per-window predictions
    windows = []
    for fc in backtest:
        vals = fc.values().flatten()
        if len(vals) == HORIZON:
            windows.append(vals)
    return np.array(windows)


def diebold_mariano(errors1, errors2, h=1):
    """
    Diebold-Mariano test for equal predictive accuracy.
    Uses squared errors as loss function.
    H0: E[L(e1)] = E[L(e2)]
    Returns: DM statistic, p-value (two-sided)
    """
    d = errors1**2 - errors2**2
    n = len(d)
    d_mean = d.mean()
    # Newey-West variance with h-1 lags
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        gamma_sum += 2 * gamma_k
    var_d = (gamma_0 + gamma_sum) / n
    if var_d <= 0:
        var_d = gamma_0 / n
    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
    return dm_stat, p_value


def bootstrap_mape_ci(actual_windows, pred_windows, n_bootstrap=N_BOOTSTRAP,
                      confidence=CONFIDENCE):
    """Bootstrap CI on MAPE computed over windows."""
    n_windows = len(actual_windows)
    mapes = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_windows, size=n_windows)
        a = actual_windows[idx].flatten()
        p = pred_windows[idx].flatten()
        mape = np.mean(np.abs((a - p) / a)) * 100
        mapes.append(mape)
    mapes = np.array(mapes)
    alpha = 1 - confidence
    lo = np.percentile(mapes, 100 * alpha / 2)
    hi = np.percentile(mapes, 100 * (1 - alpha / 2))
    return np.mean(mapes), lo, hi


def main():
    parser = argparse.ArgumentParser(
        description='Statistical comparison: Chronos-2 vs N-BEATS')
    parser.add_argument('--subsystem', type=str, default='SE',
                        choices=['SE', 'S', 'NE', 'N'])
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--nbeats-seeds', nargs='+', type=int,
                        default=[42, 123, 7])
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
    test_len = test_hours

    # Actual values per window
    actual_windows = []
    for i in range(0, test_len, HORIZON):
        if i + HORIZON > test_len:
            break
        actual_windows.append(full_load[test_start_idx + i : test_start_idx + i + HORIZON])
    actual_windows = np.array(actual_windows)
    n_windows = len(actual_windows)
    print(f'Test windows: {n_windows} (each {HORIZON}h)')

    # ---- Chronos-2 ----
    print(f'\n{"="*60}')
    print(f'  Running Chronos-2 (deterministic)')
    print(f'{"="*60}')
    chronos_windows = run_chronos_windows(full_load, test_start_idx, test_len, device)
    n_chronos = min(n_windows, len(chronos_windows))

    # ---- N-BEATS (multiple seeds) ----
    nbeats_all_windows = {}
    for seed in args.nbeats_seeds:
        print(f'\n{"="*60}')
        print(f'  Running N-BEATS (seed={seed})')
        print(f'{"="*60}')
        nbeats_windows = run_nbeats_windows(
            full_load, test_start_idx, test_len, args.subsystem,
            input_length=168, lr=5e-4, seed=seed,
        )
        nbeats_all_windows[seed] = nbeats_windows

    # Align window counts
    min_windows = min(n_chronos, *[len(w) for w in nbeats_all_windows.values()])
    actual_w = actual_windows[:min_windows]
    chronos_w = chronos_windows[:min_windows]
    print(f'\nAligned to {min_windows} windows')

    # ---- Per-window MAPE ----
    chronos_window_mape = np.array([
        np.mean(np.abs((actual_w[i] - chronos_w[i]) / actual_w[i])) * 100
        for i in range(min_windows)
    ])
    chronos_window_errors = (actual_w - chronos_w).flatten()

    # ---- Bootstrap CI for Chronos-2 ----
    c_mean, c_lo, c_hi = bootstrap_mape_ci(actual_w, chronos_w)

    print(f'\n{"#"*60}')
    print(f'  CHRONOS-2 MAPE: {c_mean:.2f}% [{c_lo:.2f}%, {c_hi:.2f}%] (95% CI)')
    print(f'{"#"*60}')

    # ---- Per-seed analysis for N-BEATS ----
    results = {
        'subsystem': args.subsystem,
        'n_windows': min_windows,
        'n_bootstrap': N_BOOTSTRAP,
        'chronos2': {
            'mape_mean': round(c_mean, 4),
            'mape_ci_lo': round(c_lo, 4),
            'mape_ci_hi': round(c_hi, 4),
            'note': 'deterministic inference, CI from bootstrap over test windows',
        },
        'nbeats_per_seed': {},
        'dm_tests': {},
    }

    nbeats_mapes = []
    for seed, nbeats_w in nbeats_all_windows.items():
        nbeats_w = nbeats_w[:min_windows]
        nb_mean, nb_lo, nb_hi = bootstrap_mape_ci(actual_w, nbeats_w)
        nbeats_mapes.append(nb_mean)

        # Per-window errors for DM test
        nbeats_errors = (actual_w - nbeats_w).flatten()

        # Diebold-Mariano test (window-level)
        chronos_sq = np.array([np.mean((actual_w[i] - chronos_w[i])**2)
                               for i in range(min_windows)])
        nbeats_sq = np.array([np.mean((actual_w[i] - nbeats_w[i])**2)
                              for i in range(min_windows)])
        dm_stat, dm_p = diebold_mariano(
            chronos_sq**0.5, nbeats_sq**0.5, h=1)

        # Paired t-test on per-window MAPE
        nbeats_window_mape = np.array([
            np.mean(np.abs((actual_w[i] - nbeats_w[i]) / actual_w[i])) * 100
            for i in range(min_windows)
        ])
        t_stat, t_p = stats.ttest_rel(chronos_window_mape, nbeats_window_mape)

        print(f'\n{"#"*60}')
        print(f'  N-BEATS (seed={seed}) MAPE: {nb_mean:.2f}%'
              f' [{nb_lo:.2f}%, {nb_hi:.2f}%] (95% CI)')
        print(f'  DM test vs Chronos-2: stat={dm_stat:.3f}, p={dm_p:.4f}')
        print(f'  Paired t-test (MAPE): t={t_stat:.3f}, p={t_p:.4f}')
        print(f'{"#"*60}')

        results['nbeats_per_seed'][str(seed)] = {
            'mape_mean': round(nb_mean, 4),
            'mape_ci_lo': round(nb_lo, 4),
            'mape_ci_hi': round(nb_hi, 4),
        }
        results['dm_tests'][str(seed)] = {
            'dm_statistic': round(dm_stat, 4),
            'dm_p_value': round(dm_p, 4),
            'paired_t_statistic': round(t_stat, 4),
            'paired_t_p_value': round(t_p, 4),
            'interpretation': (
                'significant (p<0.05)' if dm_p < 0.05
                else 'not significant (p>=0.05)'
            ),
        }

    # ---- Aggregate N-BEATS ----
    nb_agg_mean = np.mean(nbeats_mapes)
    nb_agg_std = np.std(nbeats_mapes)
    results['nbeats_aggregate'] = {
        'mape_mean': round(nb_agg_mean, 4),
        'mape_std': round(nb_agg_std, 4),
        'seeds': args.nbeats_seeds,
    }

    # ---- Summary ----
    print(f'\n{"="*60}')
    print(f'  SUMMARY: {args.subsystem}, {HORIZON}h')
    print(f'{"="*60}')
    print(f'  Chronos-2:  {c_mean:.2f}% [{c_lo:.2f}%, {c_hi:.2f}%]')
    print(f'  N-BEATS:    {nb_agg_mean:.2f}% ± {nb_agg_std:.2f}%')
    for seed in args.nbeats_seeds:
        dm = results['dm_tests'][str(seed)]
        print(f'    seed={seed}: DM p={dm["dm_p_value"]:.4f}'
              f' ({dm["interpretation"]})')

    ci_overlap = c_hi > min(r['mape_ci_lo']
                            for r in results['nbeats_per_seed'].values())
    print(f'\n  95% CI overlap: {"yes" if ci_overlap else "no"}')
    if ci_overlap:
        print('  → CIs overlap; difference may not be significant')
    else:
        print('  → CIs do not overlap; difference is significant')

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(
        OUTPUT_DIR, f'statistical_comparison_{args.subsystem}_{HORIZON}h.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results: {out_path}')


if __name__ == '__main__':
    main()
