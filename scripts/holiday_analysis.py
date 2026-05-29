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
# States covered by each ONS subsystem (dominant load centres).
# Used to build a subsystem-refined holiday set including state-level
# holidays. Municipal holidays for each state's capital are appended
# separately in _municipal_capitals_for_subsystem.
SUBSYSTEM_STATES = {
    'SE': ('SP', 'RJ', 'MG', 'ES'),   # + Centro-Oeste: GO, DF (skipped here)
    'S':  ('RS', 'SC', 'PR'),
    'NE': ('BA', 'PE', 'CE', 'PI', 'MA', 'RN', 'PB', 'SE', 'AL'),
    'N':  ('AM', 'PA', 'TO'),
}


def _municipal_capitals_for_subsystem(subsystem, years):
    """Return a set of dates for the main municipal holidays (capital
    city anniversaries) of the states in the subsystem. These are not
    covered by the `holidays` library.
    """
    capitals = {
        'SE': [(1, 25),  # São Paulo aniversário
               (3, 1),   # Rio de Janeiro aniversário
               (12, 12)],  # Belo Horizonte aniversário
        'S':  [(3, 26),  # Porto Alegre
               (3, 29)], # Curitiba
        'NE': [(3, 29),  # Salvador (fundação)
               (3, 8)],  # Recife (fundação)
        'N':  [(10, 24)],  # Manaus
    }
    out = set()
    for m, d in capitals.get(subsystem, []):
        for y in years:
            out.add(pd.Timestamp(y, m, d).date())
    return out


# Population shares by subsystem, IBGE 2022 approx (in millions).
# TODO: refine later with ONS load-share per state (more accurate for
# "fraction of subsystem demand affected"), and add major municipalities
# beyond the capital (Campinas, Niterói, etc).
SUBSYSTEM_TOTAL_POP_M = {'SE': 107.0, 'S': 30.4, 'NE': 57.0, 'N': 9.9}
STATE_POP_M = {
    'SP': 46.0, 'RJ': 17.2, 'MG': 21.4, 'ES': 4.0,
    'GO': 7.1, 'DF': 2.9, 'MT': 3.8, 'MS': 2.9,
    'RS': 11.4, 'SC': 7.8, 'PR': 11.5,
    'BA': 14.1, 'PE': 9.1, 'CE': 8.8, 'PI': 3.3, 'MA': 6.8,
    'RN': 3.3, 'PB': 4.0, 'SE': 2.2, 'AL': 3.1,
    'AM': 4.0, 'PA': 8.1, 'TO': 1.6,
}
CAPITAL_POP_M = {
    'SP': 12.4, 'RJ': 6.7, 'MG': 2.3, 'ES': 0.32,
    'RS': 1.33, 'SC': 0.51, 'PR': 1.83,
    'BA': 2.42, 'PE': 1.49,
    'AM': 2.06, 'PA': 1.30, 'TO': 0.31,
}


def build_holiday_weights(years, subsystem):
    """Return {date: weight ∈ (0, 1]} — weight is the fraction of the
    subsystem's population affected by any holiday falling on that date.

    - Federal public holidays & Carnaval     → weight 1.0
    - State holiday (state X only)           → pop(X) / pop(subsystem)
    - Capital-city anniversary (state X)     → pop(capital_X) / pop(subsystem)
    When multiple holidays overlap, the max weight wins (affected share
    is bounded by 1, not additive).
    """
    import holidays
    drop_names = {'Véspera de Natal', 'Véspera de Ano-Novo',
                  'Início da Quaresma', 'Dia do Servidor Público'}
    total = SUBSYSTEM_TOTAL_POP_M.get(subsystem)
    if total is None:
        raise ValueError(f'unknown subsystem {subsystem}')

    weights = {}

    def bump(d, w):
        prev = weights.get(d, 0.0)
        if w > prev:
            weights[d] = w

    # federal
    br_fed = holidays.Brazil(years=years, categories=('public', 'optional'))
    for d, n in br_fed.items():
        if n in drop_names:
            continue
        bump(pd.Timestamp(d).date(), 1.0)

    # state holidays (restrict to this state's extra dates,
    # i.e. dates NOT already in the federal set for that state's subdiv)
    fed_dates = {pd.Timestamp(d).date() for d, n in br_fed.items()
                 if n not in drop_names}
    for sd in SUBSYSTEM_STATES.get(subsystem, ()):
        h = holidays.Brazil(
            years=years, subdiv=sd, categories=('public', 'optional'))
        w = STATE_POP_M.get(sd, 0.0) / total
        for d, n in h.items():
            if n in drop_names:
                continue
            d0 = pd.Timestamp(d).date()
            # if already federal, skip — federal weight is already 1.0
            if d0 in fed_dates:
                continue
            bump(d0, w)

    # municipal (capital anniversaries only, hardcoded)
    _municipal = {
        'SE': [(1, 25, 'SP'), (3, 1, 'RJ'), (12, 12, 'MG')],
        'S':  [(3, 26, 'RS'), (3, 29, 'PR')],
        'NE': [(3, 29, 'BA'), (3, 8, 'PE')],
        'N':  [(10, 24, 'AM')],
    }
    for m, d, st in _municipal.get(subsystem, []):
        cap_pop = CAPITAL_POP_M.get(st, 0.0)
        w = cap_pop / total
        for y in years:
            bump(pd.Timestamp(y, m, d).date(), w)

    return weights


def build_br_holiday_set(years, subsystem=None, refined=False):
    """Return (holiday_dates, carnaval_dates) as two sets of date objects.

    If refined=False: federal public + optional (minus half-day eves).
    If refined=True and subsystem given: also unions state-level holidays
    for the subsystem's states and adds capital-city anniversaries.
    """
    import holidays
    drop_names = {'Véspera de Natal', 'Véspera de Ano-Novo',
                  'Início da Quaresma', 'Dia do Servidor Público'}

    br_fed = holidays.Brazil(years=years, categories=('public', 'optional'))
    br_fed = {d: n for d, n in br_fed.items() if n not in drop_names}
    holiday_dates = {pd.Timestamp(d).date() for d in br_fed.keys()}
    carnaval_dates = {pd.Timestamp(d).date()
                      for d, n in br_fed.items() if n == 'Carnaval'}

    if refined and subsystem in SUBSYSTEM_STATES:
        for sd in SUBSYSTEM_STATES[subsystem]:
            st = holidays.Brazil(
                years=years, subdiv=sd, categories=('public', 'optional'))
            st = {d: n for d, n in st.items() if n not in drop_names}
            holiday_dates |= {pd.Timestamp(d).date() for d in st.keys()}
        holiday_dates |= _municipal_capitals_for_subsystem(subsystem, years)

    return holiday_dates, carnaval_dates


def build_holiday_features(dates, subsystem=None, refined=False):
    """Classify each hour in `dates` into one of CATEGORIES.

    subsystem + refined: if True, expand the holiday set to include
    state-level and capital-city-municipal holidays for the subsystem.
    """
    years = sorted({pd.Timestamp(d).year for d in dates})
    holiday_dates, carnaval_days = build_br_holiday_set(
        years, subsystem=subsystem, refined=refined)

    # Expose a dict-like view for the caller (name not critical, just
    # used for the "holidays detected" count).
    br = {d: 'holiday' for d in holiday_dates}

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
