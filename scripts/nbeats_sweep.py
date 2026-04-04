#!/usr/bin/env python3
"""
N-BEATS hyperparameter sweep to address reviewer W2 concern
that N-BEATS may be under-tuned.

Grid search over input_length × learning_rate, then run the best
configuration 3× with different seeds to report mean ± std.

Usage:
    python scripts/nbeats_sweep.py                    # SE, 24h (default)
    python scripts/nbeats_sweep.py --subsystem NE     # other subsystem
    python scripts/nbeats_sweep.py --skip-sweep       # skip grid, just run seeds on best known config
"""

import argparse
import itertools
import os
import subprocess
import sys
import json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results')
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, 'train_nbeats.py')

# Hyperparameter grid
INPUT_LENGTHS = [168, 336, 720]
LEARNING_RATES = [1e-4, 5e-4, 1e-3]
SEEDS = [42, 123, 7]


def run_nbeats(subsystem, horizon, input_length, lr, seed, epochs=200,
               patience=10):
    """Run train_nbeats.py as a subprocess and parse metrics from output."""
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        '--subsystem', subsystem,
        '--horizon', str(horizon),
        '--input-length', str(input_length),
        '--lr', str(lr),
        '--random-state', str(seed),
        '--epochs', str(epochs),
        '--patience', str(patience),
    ]
    print(f'\n{"="*70}')
    print(f'  Running: input_length={input_length}, lr={lr}, seed={seed}')
    print(f'  Command: {" ".join(cmd)}')
    print(f'{"="*70}\n')

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f'STDERR:\n{result.stderr}')
        return None

    # Parse metrics from stdout
    metrics = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        for metric_name in ['MAE (MW)', 'RMSE (MW)', 'MAPE', 'MASE', 'RMSSE', 'R2']:
            if metric_name in line:
                # Format: "  MAE (MW):  829.30"  or "  MAPE:  1.86%"
                val_str = line.split(':')[-1].strip().rstrip('%')
                try:
                    metrics[metric_name] = float(val_str)
                except ValueError:
                    pass
    return metrics if metrics else None


def main():
    parser = argparse.ArgumentParser(
        description='N-BEATS hyperparameter sweep (W2)')
    parser.add_argument('--subsystem', type=str, default='SE',
                        choices=['SE', 'S', 'NE', 'N'])
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--skip-sweep', action='store_true',
                        help='Skip grid search, run seeds on best known config')
    parser.add_argument('--best-input-length', type=int, default=None,
                        help='Override best input_length for seed runs')
    parser.add_argument('--best-lr', type=float, default=None,
                        help='Override best lr for seed runs')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sweep_csv = os.path.join(
        OUTPUT_DIR, f'nbeats_sweep_{args.subsystem}_{args.horizon}h.csv')

    # ------------------------------------------------------------------
    # Phase 1: Grid search (single seed=42)
    # ------------------------------------------------------------------
    if not args.skip_sweep:
        print(f'\n{"#"*70}')
        print(f'  PHASE 1: Grid search over input_length × lr')
        print(f'  Configs: {len(INPUT_LENGTHS)} × {len(LEARNING_RATES)}'
              f' = {len(INPUT_LENGTHS) * len(LEARNING_RATES)} runs')
        print(f'{"#"*70}')

        sweep_results = []
        for input_len, lr in itertools.product(INPUT_LENGTHS, LEARNING_RATES):
            metrics = run_nbeats(
                args.subsystem, args.horizon, input_len, lr, seed=42)
            if metrics:
                row = {'input_length': input_len, 'lr': lr, 'seed': 42}
                row.update(metrics)
                sweep_results.append(row)

        sweep_df = pd.DataFrame(sweep_results)
        sweep_df.to_csv(sweep_csv, index=False)
        print(f'\nSweep results saved: {sweep_csv}')
        print(f'\n{sweep_df.to_string(index=False)}')

        # Find best config by MAPE
        best_idx = sweep_df['MAPE'].idxmin()
        best_input = int(sweep_df.loc[best_idx, 'input_length'])
        best_lr = float(sweep_df.loc[best_idx, 'lr'])
        best_mape = sweep_df.loc[best_idx, 'MAPE']
        print(f'\nBest config: input_length={best_input}, lr={best_lr}'
              f' (MAPE={best_mape:.2f}%)')
    else:
        # Use overrides or defaults
        best_input = args.best_input_length or 168
        best_lr = args.best_lr or 1e-4
        print(f'\nSkipping sweep. Using: input_length={best_input},'
              f' lr={best_lr}')

    # Allow CLI overrides for phase 2
    if args.best_input_length:
        best_input = args.best_input_length
    if args.best_lr:
        best_lr = args.best_lr

    # ------------------------------------------------------------------
    # Phase 2: Run best config 3× with different seeds
    # ------------------------------------------------------------------
    print(f'\n{"#"*70}')
    print(f'  PHASE 2: Best config × 3 seeds')
    print(f'  Config: input_length={best_input}, lr={best_lr}')
    print(f'  Seeds: {SEEDS}')
    print(f'{"#"*70}')

    seed_results = []
    for seed in SEEDS:
        metrics = run_nbeats(
            args.subsystem, args.horizon, best_input, best_lr, seed=seed)
        if metrics:
            row = {'input_length': best_input, 'lr': best_lr, 'seed': seed}
            row.update(metrics)
            seed_results.append(row)

    seed_df = pd.DataFrame(seed_results)
    seed_csv = os.path.join(
        OUTPUT_DIR,
        f'nbeats_seeds_{args.subsystem}_{args.horizon}h.csv')
    seed_df.to_csv(seed_csv, index=False)

    # ------------------------------------------------------------------
    # Summary: mean ± std
    # ------------------------------------------------------------------
    print(f'\n{"#"*70}')
    print(f'  FINAL RESULTS: N-BEATS ({args.subsystem}, {args.horizon}h)')
    print(f'  Config: input_length={best_input}, lr={best_lr}')
    print(f'{"#"*70}')
    print(f'\nPer-seed results:')
    print(seed_df.to_string(index=False))

    metric_cols = ['MAE (MW)', 'RMSE (MW)', 'MAPE', 'MASE', 'RMSSE', 'R2']
    summary = {}
    for col in metric_cols:
        if col in seed_df.columns:
            mean = seed_df[col].mean()
            std = seed_df[col].std()
            summary[col] = f'{mean:.2f} +/- {std:.2f}'
    print(f'\nMean +/- std (n={len(seed_results)}):')
    for k, v in summary.items():
        pct = '%' if k == 'MAPE' else ''
        print(f'  {k:>12s}: {v}{pct}')

    # Save summary
    summary_path = os.path.join(
        OUTPUT_DIR,
        f'nbeats_final_{args.subsystem}_{args.horizon}h.json')
    summary_data = {
        'subsystem': args.subsystem,
        'horizon': args.horizon,
        'best_input_length': best_input,
        'best_lr': best_lr,
        'seeds': SEEDS,
        'n_runs': len(seed_results),
        'metrics_mean_std': summary,
        'per_seed': seed_results,
    }
    # Also include sweep results if available
    if not args.skip_sweep and os.path.exists(sweep_csv):
        summary_data['sweep_csv'] = sweep_csv

    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f'\nSummary JSON: {summary_path}')
    print(f'Seed CSV: {seed_csv}')
    if not args.skip_sweep:
        print(f'Sweep CSV: {sweep_csv}')


if __name__ == '__main__':
    main()
