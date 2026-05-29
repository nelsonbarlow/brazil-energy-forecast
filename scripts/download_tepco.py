#!/usr/bin/env python3
"""
Download hourly electricity demand data from TEPCO (Tokyo Electric Power Company).

TEPCO serves the Tokyo area (~45M people, ~40GW peak) — the largest single grid
in Japan and one of the largest in the world.

Data source: https://www.tepco.co.jp/forecast/html/juyo-YYYY-j.csv
License: Public disclosure (公開情報), free to use for research.
Granularity: 30-minute intervals published daily, going back to 2012.

Usage:
    python scripts/download_tepco.py                     # download 2019-2024
    python scripts/download_tepco.py --start 2016 --end 2024
    python scripts/download_tepco.py --start 2023 --end 2024 --resample hourly
"""

import argparse
import os
import sys
import requests
import pandas as pd
from io import StringIO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(REPO_ROOT, 'data', 'raw_tepco')
PROCESSED_DIR = os.path.join(REPO_ROOT, 'data', 'processed')

# TEPCO publishes one CSV per year. Encoding: Shift-JIS.
# Line 1: update timestamp, Line 2: blank, Line 3: headers
# Columns: DATE, TIME, 実績(万kW)  [万kW = 10 MW → multiply by 10 for MW]
# Resolution: hourly. Available: 2016–present.
TEPCO_URL = 'https://www.tepco.co.jp/forecast/html/images/juyo-{year}.csv'


def download_year(year: int) -> str | None:
    url = TEPCO_URL.format(year=year)
    out_path = os.path.join(RAW_DIR, f'tepco_juyo_{year}.csv')

    if os.path.exists(out_path):
        print(f'  {year}: already downloaded')
        return out_path

    print(f'  {year}: downloading from {url}...')
    try:
        resp = requests.get(url, timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
    except requests.RequestException as e:
        print(f'  {year}: FAILED ({e})')
        return None

    if resp.status_code != 200:
        print(f'  {year}: FAILED (HTTP {resp.status_code})')
        return None

    with open(out_path, 'wb') as f:
        f.write(resp.content)

    size_kb = len(resp.content) / 1024
    print(f'  {year}: OK ({size_kb:.0f} KB)')
    return out_path


def parse_tepco_csv(path: str) -> pd.DataFrame | None:
    """Parse a TEPCO annual demand CSV.

    Format: Shift-JIS, line 1 = update timestamp, line 2 = blank,
    line 3 = headers (DATE,TIME,実績(万kW)), then hourly rows.
    Units: 万kW (10,000 kW) → multiply by 10 to get MW.
    """
    try:
        raw = open(path, encoding='shift-jis').read()
    except UnicodeDecodeError:
        try:
            raw = open(path, encoding='cp932').read()
        except UnicodeDecodeError:
            print(f'  Could not decode {path}')
            return None

    lines = raw.splitlines()

    # Skip preamble lines until we hit the DATE header
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('DATE'):
            data_start = i
            break

    df = pd.read_csv(StringIO('\n'.join(lines[data_start:])), header=0)
    df.columns = [c.strip() for c in df.columns]

    if 'DATE' not in df.columns:
        print(f'  Unexpected columns in {path}: {list(df.columns)}')
        return None

    # Demand column: 実績(万kW) — units are 万kW, convert to MW
    demand_col = next((c for c in df.columns if '実績' in c), None)
    if demand_col is None:
        print(f'  No demand column found in {path}: {list(df.columns)}')
        return None

    time_str = df['TIME'].astype(str).str.strip() if 'TIME' in df.columns else '0:00'
    df['datetime'] = pd.to_datetime(
        df['DATE'].astype(str).str.strip() + ' ' + time_str,
        errors='coerce',
    )

    df['demand_mw'] = (
        pd.to_numeric(df[demand_col].astype(str).str.replace(',', ''), errors='coerce')
        * 10  # 万kW → MW
    )

    df = df[['datetime', 'demand_mw']].dropna()
    return df.sort_values('datetime').reset_index(drop=True)


def process_data(start_year: int, end_year: int) -> pd.DataFrame | None:
    frames = []
    for year in range(start_year, end_year + 1):
        path = os.path.join(RAW_DIR, f'tepco_juyo_{year}.csv')
        if not os.path.exists(path):
            continue
        df = parse_tepco_csv(path)
        if df is not None and not df.empty:
            frames.append(df)
        else:
            print(f'  {year}: parse failed or empty')

    if not frames:
        print('No data parsed.')
        return None

    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates('datetime').sort_values('datetime').reset_index(drop=True)


def save_processed(df: pd.DataFrame, start_year: int, end_year: int) -> str:
    base = f'tepco_hourly_load_{start_year}_{end_year}'
    parquet_path = os.path.join(PROCESSED_DIR, f'{base}.parquet')
    csv_path = os.path.join(PROCESSED_DIR, f'{base}.csv')

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f'\nSaved:')
    print(f'  {parquet_path} ({os.path.getsize(parquet_path) / 1024:.0f} KB)')
    print(f'  {csv_path} ({os.path.getsize(csv_path) / 1024:.0f} KB)')
    return csv_path


def print_summary(df: pd.DataFrame):
    print(f'\n{"="*60}')
    print(f'  Dataset Summary — TEPCO (Tokyo)')
    print(f'{"="*60}')
    print(f'  Rows:        {len(df):,}')
    print(f'  Date range:  {df["datetime"].min()} to {df["datetime"].max()}')
    print(f'  Demand (MW): mean={df["demand_mw"].mean():,.0f}, '
          f'min={df["demand_mw"].min():,.0f}, max={df["demand_mw"].max():,.0f}')
    print(f'  Missing:     {df["demand_mw"].isna().sum()}')
    print(f'  Zeros:       {(df["demand_mw"] == 0).sum()}')


def main():
    parser = argparse.ArgumentParser(description='Download TEPCO hourly electricity demand')
    parser.add_argument('--start', type=int, default=2019)
    parser.add_argument('--end', type=int, default=2024)
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f'Downloading TEPCO demand data ({args.start}–{args.end})...')
    for year in range(args.start, args.end + 1):
        download_year(year)

    print('\nParsing...')
    df = process_data(args.start, args.end)
    if df is None:
        sys.exit(1)

    print_summary(df)
    save_processed(df, args.start, args.end)


if __name__ == '__main__':
    main()
