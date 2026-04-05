#!/usr/bin/env python3
"""
Context-length sweep: how does Chronos-2's MAPE on Carnaval, holidays,
and overall change as context length grows from 30 days (720h) to
~1 year (8192h, model max)?

For each ctx length we run:
  - Chronos-2 zero-shot, no covariates
  - Chronos-2 + holiday covariates (binary is_holiday + days_to_next)

Predictions are cached per-ctx in results/preds_cov_{subsystem}_{horizon}h_{label}/.

The question: does giving Chronos-2 a prior-year Carnaval (or other
annual events) in its context window change how much the covariate
flag buys us? If yes → "context length and covariates compound" is
the paper story.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from benchmark import load_data  # noqa: E402
from holiday_analysis import build_holiday_features, conditional_mape, CATEGORIES  # noqa: E402
from holiday_covariates import build_covariates  # noqa: E402

OUTPUT_DIR = os.path.join(REPO_ROOT, 'results')

DEFAULT_CTX_LENGTHS = [720, 1440, 2880, 4320, 6000, 7200, 8192]


def run_one(model, full_load, cov_arrays, test_start_idx, test_len,
            ctx_len, horizon, use_cov, desc):
    """Single rolling prediction pass. Returns preds array."""
    preds = []
    for i in tqdm(range(0, test_len, horizon), desc=desc):
        start = test_start_idx + i - ctx_len
        stop = test_start_idx + i
        end = stop + horizon
        target = full_load[start:stop].astype(np.float32)
        actual_horizon = min(horizon, test_len - i)
        if use_cov:
            past_cov = {k: v[start:stop] for k, v in cov_arrays.items()}
            future_cov = {k: v[stop:end][:actual_horizon]
                          for k, v in cov_arrays.items()}
            inp = [{'target': target,
                    'past_covariates': past_cov,
                    'future_covariates': future_cov}]
        else:
            inp = [{'target': target}]
        out = model.predict(inp, prediction_length=actual_horizon)
        f = out[0]
        median_idx = f.shape[1] // 2
        preds.extend(f[0, median_idx, :actual_horizon].tolist())
    return np.array(preds[:test_len])


def cached_or_run(cache_path, runner_fn):
    if os.path.exists(cache_path):
        return np.load(cache_path)['preds']
    preds = runner_fn()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, preds=preds)
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subsystem', type=str, default='SE')
    p.add_argument('--horizon', type=int, default=24)
    p.add_argument('--test-year', type=int, default=2024)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--ctx-lengths', type=int, nargs='+',
                   default=DEFAULT_CTX_LENGTHS)
    args = p.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Subsystem: {args.subsystem}  Device: {device}')
    print(f'ctx lengths: {args.ctx_lengths}')

    # Data
    df, train_df, test_df = load_data(args.subsystem, 365, args.test_year)
    full_load = df['load_mw'].values.astype(np.float32)
    full_dates = pd.to_datetime(df['datetime'].values)
    test_start_idx = len(train_df)
    test_actuals = full_load[test_start_idx:]
    test_dates = full_dates[test_start_idx:]
    test_len = len(test_actuals)
    test_label = str(args.test_year)
    print(f'Test: {test_len:,} hours ({test_label})')

    # Covariates (binary, federal only — the clean version)
    cov = build_covariates(full_dates, weighted=False)
    # Categories for conditional MAPE
    categories, _ = build_holiday_features(test_dates)

    # Cache dir
    cache_dir = os.path.join(
        OUTPUT_DIR,
        f'preds_cov_{args.subsystem}_{args.horizon}h_{test_label}')

    # Load the Chronos-2 model once
    print('Loading Chronos-2...')
    from chronos import BaseChronosPipeline
    model = BaseChronosPipeline.from_pretrained(
        'amazon/chronos-2', device_map=device, dtype=torch.float32)

    # Alt cache for ctx=720 no-cov is in results/preds_{subsystem}_{horizon}h_{label}/chronos.npz
    alt_720_nocov = os.path.join(
        OUTPUT_DIR,
        f'preds_{args.subsystem}_{args.horizon}h_{test_label}',
        'chronos.npz')

    rows = []
    for ctx in args.ctx_lengths:
        # no-cov path
        nocov_cache = os.path.join(
            cache_dir, f'chronos_nocov_ctx{ctx}.npz')
        if ctx == 720 and os.path.exists(alt_720_nocov) and not os.path.exists(nocov_cache):
            # reuse the H1 cache
            print(f'ctx={ctx}  no-cov: reusing H1 cache {alt_720_nocov}')
            nocov_preds = np.load(alt_720_nocov)['preds']
        else:
            nocov_preds = cached_or_run(
                nocov_cache,
                lambda c=ctx: run_one(
                    model, full_load, cov, test_start_idx, test_len,
                    c, args.horizon, use_cov=False,
                    desc=f'ctx={c} no-cov'))

        # cov path
        cov_cache = os.path.join(
            cache_dir, f'chronos_cov_base_ctx{ctx}.npz')
        cov_preds = cached_or_run(
            cov_cache,
            lambda c=ctx: run_one(
                model, full_load, cov, test_start_idx, test_len,
                c, args.horizon, use_cov=True,
                desc=f'ctx={c} +cov'))

        # per-category MAPE for both
        nocov_m = conditional_mape(test_actuals, nocov_preds, categories)
        cov_m = conditional_mape(test_actuals, cov_preds, categories)

        for cat in ['overall', 'normal', 'holiday', 'carnaval',
                    'day_before', 'day_after', 'bridge']:
            rows.append({
                'ctx': ctx, 'category': cat,
                'n_hours': nocov_m[cat][1],
                'nocov_mape': nocov_m[cat][0],
                'cov_mape': cov_m[cat][0],
                'delta_pp': cov_m[cat][0] - nocov_m[cat][0],
                'delta_pct': (100*(cov_m[cat][0] - nocov_m[cat][0])
                              / nocov_m[cat][0]
                              if nocov_m[cat][0] > 0 else np.nan),
            })

    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(
        OUTPUT_DIR,
        f'ctx_sweep_{args.subsystem}_{args.horizon}h_{test_label}.csv')
    df_out.to_csv(out_csv, index=False)
    print(f'\nCSV: {out_csv}')

    # Pretty tables — MAPE curves
    for metric in ['nocov_mape', 'cov_mape']:
        print(f'\n{"="*75}')
        print(f'  {metric} (%)')
        print(f'{"="*75}')
        pivot = df_out.pivot(index='ctx', columns='category', values=metric)
        pivot = pivot[['overall', 'normal', 'holiday', 'carnaval',
                       'day_before', 'day_after', 'bridge']]
        print(pivot.round(2).to_string())

    # Delta table
    print(f'\n{"="*75}')
    print('  Covariate gain (cov - nocov, pp)')
    print(f'{"="*75}')
    pivot = df_out.pivot(index='ctx', columns='category', values='delta_pp')
    pivot = pivot[['overall', 'normal', 'holiday', 'carnaval',
                   'day_before', 'day_after', 'bridge']]
    print(pivot.round(3).to_string())

    # Plot: overlay curves for each key category
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    cats_show = ['overall', 'holiday', 'carnaval', 'normal']
    colors = {'overall': 'k', 'holiday': '#FF5722',
              'carnaval': '#9C27B0', 'normal': '#4CAF50'}
    for ax, metric, title in zip(
            axes, ['nocov_mape', 'cov_mape'],
            ['No covariates', '+ Holiday covariates']):
        for c in cats_show:
            sub = df_out[df_out['category'] == c].sort_values('ctx')
            ax.plot(sub['ctx'], sub[metric], marker='o',
                    color=colors[c], label=c)
        ax.set_xlabel('context length (hours)')
        ax.set_ylabel('MAPE (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    plt.tight_layout()
    chart_path = os.path.join(
        OUTPUT_DIR,
        f'ctx_sweep_{args.subsystem}_{args.horizon}h_{test_label}.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f'\nChart: {chart_path}')


if __name__ == '__main__':
    main()
