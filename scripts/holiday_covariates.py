#!/usr/bin/env python3
"""
H2 pilot: do minimal holiday covariates recover FM performance on
Brazilian holidays WITHOUT retraining weights?

Covariates passed to the zero-shot model (known past + future):
  - is_holiday           : binary, 1 on any BR national holiday (incl. Carnaval)
  - days_to_next_holiday : signed int days to nearest holiday, clipped to
                           [-7, +7] then /7. Positive = holiday ahead,
                           negative = holiday just passed, 0 = today.

Only Chronos-2 is covered here (its covariate API is clean). Moirai
covariates are an extension.

Usage:
    python scripts/holiday_covariates.py                      # SE, 24h, 2024
    python scripts/holiday_covariates.py --test-year 2023
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
from holiday_analysis import (  # noqa: E402
    build_holiday_features, conditional_mape, CATEGORIES,
)

OUTPUT_DIR = os.path.join(REPO_ROOT, 'results')


# ---------------------------------------------------------------------------
# Covariate construction
# ---------------------------------------------------------------------------
def build_covariates(dates, carnaval_flag=False,
                     weighted=False, subsystem='SE'):
    """Build minimal holiday covariates over `dates` (hourly timestamps).

    Returns dict aligned to dates with:
      is_holiday            : float — binary {0, 1} (federal only) OR
                              weighted fraction ∈ [0, 1] (population
                              share of subsystem affected, when
                              weighted=True). Weighted mode includes
                              state + capital-municipal holidays for the
                              subsystem.
      days_to_next_holiday  : signed, clipped to [-7, 7], then /7
                              (computed from the dates with weight>0 in
                              weighted mode, federal-only in binary)
      is_carnaval           : {0, 1}  (optional, if carnaval_flag=True)
    """
    import holidays
    from holiday_analysis import build_holiday_weights

    dates = pd.to_datetime(dates)
    years = sorted({d.year for d in dates})
    years_ext = years + [min(years) - 1, max(years) + 1]
    br_all = holidays.Brazil(years=years_ext,
                             categories=('public', 'optional'))
    drop_names = {'Véspera de Natal', 'Véspera de Ano-Novo',
                  'Início da Quaresma', 'Dia do Servidor Público'}
    holiday_set = {pd.Timestamp(d).date()
                   for d, n in br_all.items() if n not in drop_names}
    carnaval_set = {pd.Timestamp(d).date()
                    for d, n in br_all.items() if n == 'Carnaval'}

    if weighted:
        weights = build_holiday_weights(years_ext, subsystem)
        is_holiday = np.array(
            [weights.get(d.date(), 0.0) for d in dates],
            dtype=np.float32)
        # days_to_next uses any date with positive weight
        holiday_for_dist = set(weights.keys())
    else:
        is_holiday = np.array(
            [1.0 if d.date() in holiday_set else 0.0 for d in dates],
            dtype=np.float32)
        holiday_for_dist = holiday_set

    # Signed distance in days to nearest holiday.
    # Sorted holiday list lets us binary-search nearest neighbours.
    hol_sorted = np.array(sorted(holiday_for_dist))
    day_dates = np.array([d.date() for d in dates])
    unique_days = pd.unique(day_dates)

    # compute per unique day
    signed = {}
    hol_ords = np.array([pd.Timestamp(h).toordinal() for h in hol_sorted])
    for dd in unique_days:
        d_ord = pd.Timestamp(dd).toordinal()
        idx = np.searchsorted(hol_ords, d_ord)
        candidates = []
        if idx < len(hol_ords):
            candidates.append(hol_ords[idx] - d_ord)  # future -> positive
        if idx > 0:
            candidates.append(hol_ords[idx-1] - d_ord)  # past -> negative
        # pick smallest |d|
        best = min(candidates, key=lambda x: (abs(x), -x))
        signed[dd] = best

    days_signed = np.array([signed[d] for d in day_dates], dtype=np.float32)
    days_to_next = np.clip(days_signed, -7, 7) / 7.0
    out = {
        'is_holiday': is_holiday,
        'days_to_next_holiday': days_to_next.astype(np.float32),
    }
    if carnaval_flag:
        # Carnaval regime: from the Friday before Carnaval Monday through
        # Ash Wednesday (i.e. Fri/Sat/Sun/Mon/Tue/Wed = 6-day window).
        # This captures the full "Carnaval week" load pattern, not just the
        # two official days.
        carnaval_region = set()
        for c in carnaval_set:
            c_ts = pd.Timestamp(c)  # Carnaval Monday
            for offset in range(-3, 3):  # Fri(-3) .. Wed(+2)
                carnaval_region.add((c_ts + pd.Timedelta(days=offset)).date())
        is_carnaval = np.array(
            [1.0 if d.date() in carnaval_region else 0.0 for d in dates],
            dtype=np.float32)
        out['is_carnaval'] = is_carnaval
    return out


# ---------------------------------------------------------------------------
# Chronos-2 with covariates
# ---------------------------------------------------------------------------
def run_chronos_with_covariates(full_load, cov_arrays, test_start_idx,
                                test_len, ctx_len, horizon, device,
                                use_covariates=True):
    """Zero-shot Chronos-2, optionally with holiday covariates.

    When use_covariates=False, this runs a plain zero-shot forecast at
    the same context length — the matched baseline for isolating the
    covariate-only effect from any context-length effect.
    """
    from chronos import BaseChronosPipeline
    model = BaseChronosPipeline.from_pretrained(
        'amazon/chronos-2', device_map=device, dtype=torch.float32,
    )
    desc = 'Chronos-2 + covars' if use_covariates else 'Chronos-2 (no cov)'
    preds = []
    for i in tqdm(range(0, test_len, horizon), desc=desc):
        start = test_start_idx + i - ctx_len
        stop = test_start_idx + i
        end = stop + horizon
        target = full_load[start:stop].astype(np.float32)
        actual_horizon = min(horizon, test_len - i)
        if use_covariates:
            past_cov = {k: v[start:stop] for k, v in cov_arrays.items()}
            future_cov = {k: v[stop:end][:actual_horizon]
                          for k, v in cov_arrays.items()}
            inp = [{
                'target': target,
                'past_covariates': past_cov,
                'future_covariates': future_cov,
            }]
        else:
            inp = [{'target': target}]
        out = model.predict(inp, prediction_length=actual_horizon)
        f = out[0]  # shape (n_variates=1, quantiles, pred_len)
        median_idx = f.shape[1] // 2
        preds.extend(f[0, median_idx, :actual_horizon].tolist())
    return np.array(preds[:test_len])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subsystem', type=str, default='SE',
                   choices=['SE', 'S', 'NE', 'N'])
    p.add_argument('--horizon', type=int, default=24)
    p.add_argument('--context-length', type=int, default=720)
    p.add_argument('--test-days', type=int, default=365)
    p.add_argument('--test-year', type=int, default=None)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--force-rerun', action='store_true')
    p.add_argument('--carnaval-flag', action='store_true',
                   help='Add is_carnaval as a 3rd covariate (H2b)')
    p.add_argument('--no-covariates', action='store_true',
                   help='Run plain zero-shot at this ctx_len (matched baseline)')
    p.add_argument('--weighted', action='store_true',
                   help='Use population-weighted holiday flag (includes '
                        'state+capital-municipal holidays for the subsystem)')
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
    print(f'Horizon: {args.horizon}h  Context: {args.context_length}h')

    df, train_df, test_df = load_data(
        args.subsystem, args.test_days, args.test_year)
    full_load = df['load_mw'].values.astype(np.float32)
    full_dates = pd.to_datetime(df['datetime'].values)
    test_start_idx = len(train_df)
    test_actuals = full_load[test_start_idx:]
    test_dates = full_dates[test_start_idx:]
    test_len = len(test_actuals)
    test_label = (str(args.test_year) if args.test_year
                  else f'{args.test_days}d')
    print(f'Test: {test_len:,} hours ({test_label})')

    # Build covariates over the FULL series (train+test) so the rolling
    # window can always slice past and future
    cov = build_covariates(full_dates, carnaval_flag=args.carnaval_flag,
                           weighted=args.weighted, subsystem=args.subsystem)
    print(f'Covariates: {list(cov.keys())}')
    print(f'  is_holiday=1 hours in full series: '
          f'{int(cov["is_holiday"].sum())}/{len(cov["is_holiday"])}')

    # Categories for conditional MAPE
    categories, _ = build_holiday_features(test_dates)

    # Cache dir for covariate runs (keyed by ctx_len so long-context runs
    # don't clobber the default ctx=720 ones)
    cache_dir = os.path.join(
        OUTPUT_DIR, f'preds_cov_{args.subsystem}_{args.horizon}h_{test_label}')
    os.makedirs(cache_dir, exist_ok=True)
    cov_tag = 'carnaval' if args.carnaval_flag else 'base'
    if args.weighted:
        cov_tag = cov_tag + '_weighted'
    ctx_tag = f'ctx{args.context_length}'
    if args.no_covariates:
        cache_path = os.path.join(cache_dir, f'chronos_nocov_{ctx_tag}.npz')
    else:
        cache_path = os.path.join(
            cache_dir, f'chronos_cov_{cov_tag}_{ctx_tag}.npz')

    if args.force_rerun and os.path.exists(cache_path):
        os.remove(cache_path)

    if os.path.exists(cache_path):
        print(f'Loaded cached: {cache_path}')
        chronos_cov_preds = np.load(cache_path)['preds']
    else:
        chronos_cov_preds = run_chronos_with_covariates(
            full_load, cov, test_start_idx, test_len,
            args.context_length, args.horizon, device,
            use_covariates=not args.no_covariates)
        np.savez(cache_path, preds=chronos_cov_preds)
        print(f'Cached: {cache_path}')

    # Baseline: matched-ctx no-covariate run if it exists; otherwise
    # fall back to the H1 720h cache for convenience.
    matched_baseline = os.path.join(
        cache_dir, f'chronos_nocov_{ctx_tag}.npz')
    if os.path.exists(matched_baseline) and not args.no_covariates:
        print(f'Using matched-ctx baseline: {matched_baseline}')
        chronos_zs_preds = np.load(matched_baseline)['preds']
        baseline_label = f'no-cov @ ctx={args.context_length}'
    else:
        zs_cache = os.path.join(
            OUTPUT_DIR,
            f'preds_{args.subsystem}_{args.horizon}h_{test_label}',
            'chronos.npz')
        if not os.path.exists(zs_cache):
            print(f'ERROR: baseline Chronos cache missing at {zs_cache}')
            sys.exit(1)
        chronos_zs_preds = np.load(zs_cache)['preds']
        baseline_label = 'zero-shot @ ctx=720 (H1)'

    if args.no_covariates:
        print(f'\n(--no-covariates run: predictions cached for later use as '
              f'matched baseline at ctx={args.context_length})')

    # Conditional MAPE for both
    zs_m = conditional_mape(test_actuals, chronos_zs_preds, categories)
    cov_m = conditional_mape(test_actuals, chronos_cov_preds, categories)

    # Side-by-side table
    rows = []
    for c in ['overall'] + CATEGORIES:
        zs = zs_m[c][0]
        cv = cov_m[c][0]
        delta = cv - zs
        pct = 100 * delta / zs if zs > 0 and not np.isnan(zs) else np.nan
        rows.append({'category': c, 'n_hours': zs_m[c][1],
                     'zero_shot': zs, 'with_cov': cv,
                     'delta': delta, 'pct_change': pct})
    table = pd.DataFrame(rows).set_index('category').round(3)

    print(f'\n{"="*75}')
    run_label = 'no-cov' if args.no_covariates else '+ holiday covariates'
    print(f'  Chronos-2 MAPE (%): [{baseline_label}] vs [{run_label} '
          f'@ ctx={args.context_length}]')
    print(f'{"="*75}')
    print(table.to_string())

    out_csv = os.path.join(
        OUTPUT_DIR,
        f'holiday_covariates_{args.subsystem}_{args.horizon}h_{test_label}.csv')
    table.to_csv(out_csv)
    print(f'\nCSV: {out_csv}')

    # Verdict
    print(f'\n{"="*75}')
    print('  H2 verdict (do minimal covariates recover FM on holidays?)')
    print(f'{"="*75}')
    for c in ['holiday', 'carnaval', 'bridge', 'day_before', 'day_after']:
        zs = zs_m[c][0]
        cv = cov_m[c][0]
        if zs > 0 and not np.isnan(zs):
            print(f'  {c:>12s}: {zs:5.2f}% -> {cv:5.2f}%  '
                  f'(Δ {cv-zs:+.2f}pp, {100*(cv-zs)/zs:+.1f}%)')

    # Also compare to normal-day MAPE to see if gap closes
    normal_zs = zs_m['normal'][0]
    normal_cov = cov_m['normal'][0]
    print(f'\n  reference — normal-day MAPE: '
          f'{normal_zs:.2f}% (ZS) vs {normal_cov:.2f}% (+ cov)')
    print(f'  holiday gap: {zs_m["holiday"][0]-normal_zs:+.2f}pp (ZS) '
          f'-> {cov_m["holiday"][0]-normal_cov:+.2f}pp (+ cov)')
    print(f'  carnaval gap: {zs_m["carnaval"][0]-normal_zs:+.2f}pp (ZS) '
          f'-> {cov_m["carnaval"][0]-normal_cov:+.2f}pp (+ cov)')


if __name__ == '__main__':
    main()
