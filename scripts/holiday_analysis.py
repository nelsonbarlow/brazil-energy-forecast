#!/usr/bin/env python3
"""
H1 pilot: do zero-shot FMs degrade on Brazilian holidays?

For each model, compute conditional MAPE over the test year, broken down by:
  - normal      : regular weekday, not near any holiday
  - weekend     : Saturday/Sunday, not holiday
  - holiday     : any national Brazilian holiday (incl. Carnaval Mon/Tue)
  - day_before  : day immediately before a holiday
  - day_after   : day immediately after a holiday
  - bridge      : weekday wedged between holiday and weekend (feriadão)
  - carnaval    : the full Carnaval week (Mon–Wed incl. Ash Wed)

Predictions are cached to results/preds_{subsystem}_{horizon}h/{model}.npz
so iterating on the analysis is fast after the first run.

Usage:
    python scripts/holiday_analysis.py                       # SE, 24h, all models
    python scripts/holiday_analysis.py --models naive chronos
    python scripts/holiday_analysis.py --subsystem NE
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# Reuse benchmark.py infrastructure (data loading + model runners)
from benchmark import (  # noqa: E402
    load_data, run_naive, run_chronos, run_tirex, run_moirai,
)

OUTPUT_DIR = os.path.join(REPO_ROOT, 'results')

MODEL_RUNNERS = {
    'naive':   ('Naive (7d ago)', run_naive),
    'chronos': ('Chronos-2',      run_chronos),
    'tirex':   ('TiRex',          run_tirex),
    'moirai':  ('Moirai 2.0',     run_moirai),
}

CATEGORIES = ['normal', 'weekend', 'holiday', 'day_before',
              'day_after', 'bridge', 'carnaval']


# ---------------------------------------------------------------------------
# Holiday classification
# ---------------------------------------------------------------------------
def build_holiday_features(dates):
    """Classify each hour in `dates` into one of CATEGORIES.

    Uses the `holidays` library for Brazilian national holidays, plus
    manual Carnaval Monday/Tuesday (de facto holidays) which the lib
    marks as optional.
    """
    import holidays

    years = sorted({pd.Timestamp(d).year for d in dates})
    # 'public' = official federal holidays; 'optional' adds Carnaval,
    # Corpus Christi, and eves. We filter eves/Ash-Wed since they are
    # half-days that don't shift load like full holidays do.
    br_all = holidays.Brazil(
        years=years, categories=('public', 'optional'))
    drop_names = {'Véspera de Natal', 'Véspera de Ano-Novo',
                  'Início da Quaresma', 'Dia do Servidor Público'}
    br = {d: n for d, n in br_all.items() if n not in drop_names}

    # Identify Carnaval (Mon & Tue before Ash Wednesday) separately
    # for its own analysis category — Carnaval week is the prime
    # regime-shift example in Brazilian load.
    carnaval_days = {d for d, n in br.items() if n == 'Carnaval'}

    holiday_dates = set(br.keys())

    # Build per-date category (one category per calendar day,
    # then broadcast to hours)
    unique_dates = sorted({pd.Timestamp(d).date() for d in dates})
    date_cat = {}
    for d in unique_dates:
        d_ts = pd.Timestamp(d)
        dow = d_ts.dayofweek  # 0=Mon, 6=Sun

        if d in carnaval_days:
            date_cat[d] = 'carnaval'
            continue
        if d in holiday_dates:
            date_cat[d] = 'holiday'
            continue

        # bridge day (feriadão): weekday wedged between holiday and
        # weekend — classic cases are Fri after a Thu holiday, Mon
        # before a Tue holiday. Checked BEFORE day_before/day_after
        # because bridge load patterns are distinct (many take off).
        prev_is_holiday = (d - pd.Timedelta(days=1)) in holiday_dates
        next_is_holiday = (d + pd.Timedelta(days=1)) in holiday_dates
        is_bridge = (
            (dow == 4 and prev_is_holiday) or  # Fri after Thu holiday
            (dow == 0 and next_is_holiday)     # Mon before Tue holiday
        )
        if is_bridge:
            date_cat[d] = 'bridge'
            continue

        # day_before / day_after a holiday (or Carnaval)
        if next_is_holiday or (d + pd.Timedelta(days=1)) in carnaval_days:
            date_cat[d] = 'day_before'
            continue
        if prev_is_holiday or (d - pd.Timedelta(days=1)) in carnaval_days:
            date_cat[d] = 'day_after'
            continue

        if dow >= 5:
            date_cat[d] = 'weekend'
        else:
            date_cat[d] = 'normal'

    cats = np.array([date_cat[pd.Timestamp(d).date()] for d in dates])
    return cats, br


# ---------------------------------------------------------------------------
# Prediction cache
# ---------------------------------------------------------------------------
def get_or_run_preds(model_key, full_load, test_start_idx, test_len,
                     ctx_len, horizon, device, cache_dir):
    """Load cached predictions or generate and cache them."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{model_key}.npz')

    if os.path.exists(cache_path):
        print(f'  Loaded cached predictions: {cache_path}')
        return np.load(cache_path)['preds']

    display_name, runner = MODEL_RUNNERS[model_key]
    print(f'  Running {display_name}...')
    if model_key == 'naive':
        preds = runner(full_load, test_start_idx, test_len, ctx_len, horizon)
    else:
        preds = runner(full_load, test_start_idx, test_len, ctx_len,
                       horizon, device)
    np.savez(cache_path, preds=preds)
    print(f'  Cached: {cache_path}')
    return preds


# ---------------------------------------------------------------------------
# Conditional MAPE
# ---------------------------------------------------------------------------
def conditional_mape(actual, predicted, categories):
    """MAPE per category. Returns dict category -> (mape, count)."""
    out = {}
    overall = mean_absolute_percentage_error(actual, predicted) * 100
    out['overall'] = (overall, len(actual))
    for cat in CATEGORIES:
        mask = categories == cat
        if mask.sum() == 0:
            out[cat] = (np.nan, 0)
            continue
        mape = mean_absolute_percentage_error(
            actual[mask], predicted[mask]) * 100
        out[cat] = (mape, int(mask.sum()))
    return out


def mape_lift(cat_mape, baseline='normal'):
    """Per-category MAPE lift vs baseline (ratio)."""
    base = cat_mape[baseline][0]
    return {c: (v[0] / base if not np.isnan(v[0]) and base > 0 else np.nan,
                v[0], v[1])
            for c, v in cat_mape.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='H1: zero-shot FM degradation on Brazilian holidays')
    parser.add_argument('--subsystem', type=str, default='SE',
                        choices=['SE', 'S', 'NE', 'N'])
    parser.add_argument('--models', nargs='+',
                        choices=list(MODEL_RUNNERS.keys()),
                        default=['naive', 'chronos', 'tirex', 'moirai'])
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--context-length', type=int, default=720)
    parser.add_argument('--test-days', type=int, default=365)
    parser.add_argument('--test-year', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--force-rerun', action='store_true',
                        help='Ignore cached predictions')
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Subsystem: {args.subsystem}  Device: {device}')
    print(f'Horizon: {args.horizon}h  Context: {args.context_length}h')
    print(f'Models: {", ".join(args.models)}')

    # Data
    df, train_df, test_df = load_data(
        args.subsystem, args.test_days, args.test_year)
    full_load = df['load_mw'].values
    test_start_idx = len(train_df)
    test_actuals = full_load[test_start_idx:]
    test_dates = pd.to_datetime(df['datetime'].values[test_start_idx:])
    test_len = len(test_actuals)

    test_label = (str(args.test_year) if args.test_year
                  else f'{args.test_days}d')
    print(f'Test: {test_len:,} hours ({test_label})')

    # Categories
    categories, br_holidays = build_holiday_features(test_dates)
    print(f'\n{"="*60}')
    print('  Test-window category distribution (hours)')
    print(f'{"="*60}')
    cat_counts = pd.Series(categories).value_counts().reindex(
        CATEGORIES, fill_value=0)
    for c, n in cat_counts.items():
        print(f'  {c:>12s}: {n:>5d} hours ({100*n/test_len:5.1f}%)')
    print(f'\nBrazilian holidays detected in test window: '
          f'{len([d for d in br_holidays.keys() if pd.Timestamp(d) in test_dates.normalize()])}')

    # Cache dir
    cache_dir = os.path.join(
        OUTPUT_DIR, f'preds_{args.subsystem}_{args.horizon}h_{test_label}')
    if args.force_rerun:
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)

    # Predictions
    all_preds = {}
    for m in args.models:
        print(f'\n--- {MODEL_RUNNERS[m][0]} ---')
        preds = get_or_run_preds(
            m, full_load, test_start_idx, test_len,
            args.context_length, args.horizon, device, cache_dir)
        all_preds[m] = preds

    # Conditional MAPE table
    rows = []
    for m, preds in all_preds.items():
        display = MODEL_RUNNERS[m][0]
        cat_m = conditional_mape(test_actuals, preds, categories)
        row = {'model': display}
        for c in ['overall'] + CATEGORIES:
            row[c] = cat_m[c][0]
        rows.append(row)
    table = pd.DataFrame(rows).set_index('model').round(2)

    print(f'\n{"="*80}')
    print('  Conditional MAPE (%) by category')
    print(f'{"="*80}')
    print(table.to_string())

    # Lift table: ratio vs normal-day MAPE (per model)
    lift = table.div(table['normal'], axis=0).round(2)
    print(f'\n{"="*80}')
    print('  MAPE lift vs normal days (1.00 = same as normal)')
    print(f'{"="*80}')
    print(lift.to_string())

    # Save CSVs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    suffix = f'{args.subsystem}_{args.horizon}h_{test_label}'
    table.to_csv(os.path.join(OUTPUT_DIR, f'holiday_mape_{suffix}.csv'))
    lift.to_csv(os.path.join(OUTPUT_DIR, f'holiday_lift_{suffix}.csv'))
    print(f'\nCSVs: results/holiday_mape_{suffix}.csv, holiday_lift_{suffix}.csv')

    # Plot: grouped bars, MAPE per category per model
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    cats_plot = CATEGORIES
    x = np.arange(len(cats_plot))
    width = 0.8 / len(table.index)
    colors = ['#888888', '#FF5722', '#2196F3', '#4CAF50', '#9C27B0'][:len(table.index)]
    for i, (model, row) in enumerate(table.iterrows()):
        vals = [row[c] for c in cats_plot]
        axes[0].bar(x + i*width, vals, width, label=model, color=colors[i])
    axes[0].set_xticks(x + width*(len(table)-1)/2)
    axes[0].set_xticklabels(cats_plot, rotation=30, ha='right')
    axes[0].set_ylabel('MAPE (%)')
    axes[0].set_title(f'Conditional MAPE — {args.subsystem}, {args.horizon}h, {test_label}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Lift plot
    for i, (model, row) in enumerate(lift.iterrows()):
        vals = [row[c] for c in cats_plot]
        axes[1].bar(x + i*width, vals, width, label=model, color=colors[i])
    axes[1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xticks(x + width*(len(lift)-1)/2)
    axes[1].set_xticklabels(cats_plot, rotation=30, ha='right')
    axes[1].set_ylabel('MAPE / MAPE(normal)')
    axes[1].set_title('MAPE lift vs normal days (higher = worse on that regime)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, f'holiday_analysis_{suffix}.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f'Chart: {chart_path}')

    # Quick verdict
    print(f'\n{"="*80}')
    print('  H1 verdict (does zero-shot break on holidays?)')
    print(f'{"="*80}')
    for model in table.index:
        base = table.loc[model, 'normal']
        hol = table.loc[model, 'holiday']
        car = table.loc[model, 'carnaval']
        print(f'  {model:>18s}: normal={base:.2f}%  holiday={hol:.2f}% '
              f'(x{hol/base:.2f})  carnaval={car:.2f}% (x{car/base:.2f})')


if __name__ == '__main__':
    main()
