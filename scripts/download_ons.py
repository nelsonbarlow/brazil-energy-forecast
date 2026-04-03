#!/usr/bin/env python3
"""
Download hourly load data from ONS (Operador Nacional do Sistema Eletrico).

Data source: https://dados.ons.org.br/dataset/curva-carga
License: CC-BY 4.0

Usage:
    python scripts/download_ons.py                     # download 2019-2025
    python scripts/download_ons.py --start 2000 --end 2025  # custom range
    python scripts/download_ons.py --subsystem SE      # single subsystem
"""

import argparse
import os
import sys
import requests
import pandas as pd
from io import StringIO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(REPO_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(REPO_ROOT, 'data', 'processed')

S3_BASE = 'https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/curva-carga-ho'

SUBSYSTEMS = {
    'SE': 'SUDESTE',   # Sudeste/Centro-Oeste (largest, ~55% of national load)
    'S':  'SUL',       # Sul
    'NE': 'NORDESTE',  # Nordeste
    'N':  'NORTE',     # Norte
}


def download_year(year):
    """Download one year of hourly load data from ONS S3."""
    url = f'{S3_BASE}/CURVA_CARGA_{year}.csv'
    out_path = os.path.join(RAW_DIR, f'curva_carga_{year}.csv')

    if os.path.exists(out_path):
        print(f'  {year}: already downloaded')
        return out_path

    print(f'  {year}: downloading from {url}...')
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        print(f'  {year}: FAILED (HTTP {resp.status_code})')
        return None

    with open(out_path, 'wb') as f:
        f.write(resp.content)

    size_mb = len(resp.content) / (1024 * 1024)
    print(f'  {year}: OK ({size_mb:.1f} MB)')
    return out_path


def process_data(start_year, end_year, subsystem=None):
    """Load all downloaded CSVs into a single clean DataFrame."""
    frames = []
    for year in range(start_year, end_year + 1):
        path = os.path.join(RAW_DIR, f'curva_carga_{year}.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep=';', encoding='utf-8')
        frames.append(df)

    if not frames:
        print('No data files found.')
        return None

    df = pd.concat(frames, ignore_index=True)

    # Rename columns to English
    df = df.rename(columns={
        'id_subsistema': 'subsystem',
        'nom_subsistema': 'subsystem_name',
        'din_instante': 'datetime',
        'val_cargaenergiahomwmed': 'load_mw',
    })

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['subsystem', 'datetime']).reset_index(drop=True)

    # Filter to specific subsystem if requested
    if subsystem:
        subsystem = subsystem.upper()
        df = df[df['subsystem'] == subsystem].reset_index(drop=True)

    return df


def save_processed(df, subsystem=None):
    """Save processed data as parquet and CSV."""
    suffix = f'_{subsystem.lower()}' if subsystem else '_all'
    start = df['datetime'].min().strftime('%Y')
    end = df['datetime'].max().strftime('%Y')
    base = f'ons_hourly_load{suffix}_{start}_{end}'

    parquet_path = os.path.join(PROCESSED_DIR, f'{base}.parquet')
    csv_path = os.path.join(PROCESSED_DIR, f'{base}.csv')

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f'\nSaved:')
    print(f'  {parquet_path} ({os.path.getsize(parquet_path) / 1024 / 1024:.1f} MB)')
    print(f'  {csv_path} ({os.path.getsize(csv_path) / 1024 / 1024:.1f} MB)')
    return csv_path


def print_summary(df):
    """Print dataset summary."""
    print(f'\n{"="*60}')
    print(f'  Dataset Summary')
    print(f'{"="*60}')
    print(f'  Rows:        {len(df):,}')
    print(f'  Date range:  {df["datetime"].min()} to {df["datetime"].max()}')
    print(f'  Subsystems:  {", ".join(sorted(df["subsystem"].unique()))}')
    print(f'\n  Load (MW) by subsystem:')
    for sub in sorted(df['subsystem'].unique()):
        s = df[df['subsystem'] == sub]['load_mw']
        print(f'    {sub:>2s} ({SUBSYSTEMS.get(sub, "?")}): '
              f'mean={s.mean():,.0f}, min={s.min():,.0f}, max={s.max():,.0f}')
    print(f'\n  Missing values: {df["load_mw"].isna().sum()}')
    print(f'  Zero values:    {(df["load_mw"] == 0).sum()}')


def main():
    parser = argparse.ArgumentParser(description='Download ONS hourly load data')
    parser.add_argument('--start', type=int, default=2019,
                        help='Start year (default: 2019)')
    parser.add_argument('--end', type=int, default=2025,
                        help='End year (default: 2025)')
    parser.add_argument('--subsystem', type=str, default=None,
                        choices=['SE', 'S', 'NE', 'N'],
                        help='Filter to specific subsystem')
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Download
    print(f'Downloading ONS hourly load data ({args.start}-{args.end})...')
    for year in range(args.start, args.end + 1):
        download_year(year)

    # Process
    print(f'\nProcessing...')
    df = process_data(args.start, args.end, args.subsystem)
    if df is None:
        sys.exit(1)

    print_summary(df)
    save_processed(df, args.subsystem)


if __name__ == '__main__':
    main()
