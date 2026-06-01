#!/usr/bin/env python3
"""
H3: does holiday covariates help trained N-BEATS as much as zero-shot
Chronos-2? If the foundation-model gain is LARGER, it supports the
"universality + cheap transfer beats trained-from-scratch" narrative:
N-BEATS has already memorised Brazilian holidays implicitly in its
weights from 5+ years of training, while Chronos-2 gains more because
it has to learn the flag's meaning via ICL.

We use N-BEATS's best tuned config from W2 (input_length=168, lr=5e-4)
and train 3 seeds each for:
  - no covariates (replicates the W2 baseline)
  - with past_covariates = (is_holiday, days_to_next_holiday)

N-BEATS in Darts supports past_covariates only (not future). This is
an architectural asymmetry with Chronos-2; we report it honestly as a
feature of FMs (ingest arbitrary known-future inputs at inference).

Predictions are cached per seed for fast rerun; per-category MAPE is
computed via the holiday_analysis helpers.

Usage:
    python scripts/h3_nbeats_covariates.py                          # SE, both configs, 3 seeds
    python scripts/h3_nbeats_covariates.py --seeds 42               # just one seed
    python scripts/h3_nbeats_covariates.py --config-only cov        # only the covariate runs
"""

import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results')
CACHE_DIR = os.path.join(OUTPUT_DIR, 'h3_nbeats_cache')

sys.path.insert(0, SCRIPT_DIR)
from holiday_analysis import build_holiday_features, conditional_mape, CATEGORIES  # noqa: E402
from holiday_covariates import build_covariates  # noqa: E402


# ---------------------------------------------------------------------------
def load_load_df(subsystem):
    pattern = os.path.join(
        PROCESSED_DIR, f'ons_hourly_load_{subsystem.lower()}_*.csv')
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


def build_covariate_series(full_dates):
    """Build Darts TimeSeries for past covariates (2 channels)."""
    from darts import TimeSeries
    cov_dict = build_covariates(full_dates, weighted=False)
    cov_df = pd.DataFrame({
        'datetime': full_dates,
        'is_holiday': cov_dict['is_holiday'],
        'days_to_next_holiday': cov_dict['days_to_next_holiday'],
    })
    return TimeSeries.from_dataframe(
        cov_df, time_col='datetime',
        value_cols=['is_holiday', 'days_to_next_holiday'],
        freq='h', fill_missing_dates=True)


def train_and_predict(series, train_series, val_series, test_series,
                      past_covariates, seed, input_length, lr, horizon,
                      epochs, patience):
    """Train one N-BEATS and run rolling backtest on test_series.
    Returns pred array (flattened)."""
    from darts import concatenate
    from darts.models import NBEATSModel
    from pytorch_lightning.callbacks import EarlyStopping

    early_stopper = EarlyStopping(
        monitor='val_loss', patience=patience,
        min_delta=1e-5, mode='min')

    model = NBEATSModel(
        input_chunk_length=input_length,
        output_chunk_length=horizon,
        num_stacks=30, num_blocks=1, num_layers=4, layer_widths=256,
        n_epochs=epochs, batch_size=256,
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
        save_checkpoints=False,
    )

    fit_kwargs = {
        'series': train_series,
        'val_series': val_series,
        'verbose': True,
    }
    if past_covariates is not None:
        fit_kwargs['past_covariates'] = past_covariates
        fit_kwargs['val_past_covariates'] = past_covariates
    model.fit(**fit_kwargs)

    bt_kwargs = {
        'series': series,
        'start': test_series.start_time(),
        'forecast_horizon': horizon,
        'stride': horizon,
        'retrain': False,
        'last_points_only': False,
        'verbose': True,
    }
    if past_covariates is not None:
        bt_kwargs['past_covariates'] = past_covariates
    backtest = model.historical_forecasts(**bt_kwargs)
    forecast = concatenate(backtest)
    return forecast.values().flatten()


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subsystem', type=str, default='SE')
    p.add_argument('--horizon', type=int, default=24)
    p.add_argument('--input-length', type=int, default=168)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--test-days', type=int, default=365)
    p.add_argument('--val-days', type=int, default=60)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 7])
    p.add_argument('--config-only', type=str, choices=['cov', 'nocov'],
                   default=None,
                   help='Run only one of the two configs')
    p.add_argument('--force-rerun', action='store_true')
    args = p.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f'H3: N-BEATS + covariates vs no-covariates')
    print(f'Subsystem: {args.subsystem}  config: input={args.input_length}h, lr={args.lr}')
    print(f'Seeds: {args.seeds}')

    # --- Data ---
    from darts import TimeSeries
    df = load_load_df(args.subsystem)
    series = TimeSeries.from_dataframe(
        df, time_col='datetime', value_cols='load_mw', freq='h',
        fill_missing_dates=True)
    print(f'TimeSeries length: {len(series)} hours')

    test_hours = args.test_days * args.horizon  # 24 h/day assumed below
    test_hours = args.test_days * 24
    val_hours = args.val_days * 24
    test_series = series[-test_hours:]
    pre_test = series[:-test_hours]
    val_series = pre_test[-val_hours:]
    train_series = pre_test[:-val_hours]
    print(f'Train/Val/Test: {len(train_series)}/{len(val_series)}/{len(test_series)} h')

    # --- Covariates ---
    full_dates = pd.to_datetime(series.time_index)
    past_cov_series = build_covariate_series(full_dates)
    print(f'Covariate TimeSeries: {len(past_cov_series)} h × '
          f'{past_cov_series.width} channels')

    # --- Categories for conditional MAPE ---
    test_dates = pd.to_datetime(test_series.time_index)
    categories, _ = build_holiday_features(test_dates)
    test_actuals = test_series.values().flatten()

    # --- Configs to run ---
    configs = []
    if args.config_only != 'cov':
        configs.append(('nocov', None))
    if args.config_only != 'nocov':
        configs.append(('cov', past_cov_series))

    all_preds = {}  # {(config, seed): preds}
    for cfg_name, pc in configs:
        for seed in args.seeds:
            cache = os.path.join(
                CACHE_DIR,
                f'nbeats_{args.subsystem}_{args.horizon}h_'
                f'in{args.input_length}_lr{args.lr}_{cfg_name}_seed{seed}.npz')
            if os.path.exists(cache) and not args.force_rerun:
                print(f'\n[cached] {cfg_name} seed={seed}  -> {cache}')
                preds = np.load(cache)['preds']
            else:
                print(f'\n{"="*70}')
                print(f'  Training: config={cfg_name}  seed={seed}')
                print(f'{"="*70}')
                preds = train_and_predict(
                    series, train_series, val_series, test_series,
                    pc, seed, args.input_length, args.lr, args.horizon,
                    args.epochs, args.patience)
                # align length to test
                if len(preds) > len(test_actuals):
                    preds = preds[:len(test_actuals)]
                np.savez(cache, preds=preds)
                print(f'  Cached: {cache}')
            all_preds[(cfg_name, seed)] = preds[:len(test_actuals)]

    # --- Per-seed per-category MAPE ---
    rows = []
    for (cfg, seed), preds in all_preds.items():
        cm = conditional_mape(test_actuals, preds, categories)
        row = {'config': cfg, 'seed': seed}
        for c in ['overall'] + CATEGORIES:
            row[c] = cm[c][0]
        rows.append(row)
    df_seeds = pd.DataFrame(rows)

    print(f'\n{"="*80}')
    print('  Per-seed per-category MAPE (%)')
    print(f'{"="*80}')
    print(df_seeds.round(3).to_string(index=False))

    # --- Aggregate across seeds: mean, std ---
    agg = df_seeds.groupby('config').agg(['mean', 'std']).round(3)
    print(f'\n{"="*80}')
    print('  Mean ± Std across seeds')
    print(f'{"="*80}')
    print(agg.to_string())

    # --- Delta (cov gain) ---
    if {'cov', 'nocov'}.issubset(df_seeds['config'].unique()):
        cov_mean = df_seeds[df_seeds['config'] == 'cov'].mean(
            numeric_only=True)
        nocov_mean = df_seeds[df_seeds['config'] == 'nocov'].mean(
            numeric_only=True)
        print(f'\n{"="*80}')
        print('  Covariate gain: N-BEATS (cov) - N-BEATS (no-cov)')
        print(f'{"="*80}')
        print(f'  {"category":>12s}  {"no-cov":>7s}  {"cov":>7s}  '
              f'{"delta_pp":>9s}  {"delta_%":>8s}')
        for c in ['overall', 'normal', 'holiday', 'carnaval',
                  'day_before', 'day_after', 'bridge']:
            nc = nocov_mean[c]
            co = cov_mean[c]
            d = co - nc
            dp = 100 * d / nc if nc > 0 else np.nan
            print(f'  {c:>12s}  {nc:>7.3f}  {co:>7.3f}  '
                  f'{d:+9.3f}  {dp:+8.1f}%')

    # --- Save ---
    csv_path = os.path.join(
        OUTPUT_DIR,
        f'h3_nbeats_covariates_{args.subsystem}_{args.horizon}h.csv')
    df_seeds.to_csv(csv_path, index=False)
    print(f'\nCSV: {csv_path}')


if __name__ == '__main__':
    main()
