# Brazilian Energy Load Forecasting with Foundation Models

Benchmarking time series foundation models on Brazilian electricity load forecasting using ONS (Operador Nacional do Sistema Eletrico) data.

## Research gap

No published paper with code applies modern foundation models (Chronos-2, TiRex, Moirai) to Brazilian energy data. This project fills that gap.

## Why this matters

- Brazil's grid is hydro-dependent (~65% of generation), creating unique seasonal patterns (dry/wet seasons) that differ from US/EU grids
- Energy load forecasting is non-adversarial (your prediction doesn't change demand)
- Foundation models are designed for this type of problem and are SOTA on general forecasting benchmarks
- The question: do they transfer zero-shot to Brazil's unique grid dynamics?

## Data

**Source:** ONS Open Data Portal (https://dados.ons.org.br/)

- **License:** CC-BY 4.0 (free, open access, no registration)
- **Resolution:** Hourly
- **History:** 2000-2026 (26 years)
- **Subsystems:** SE (Sudeste/Centro-Oeste), S (Sul), NE (Nordeste), N (Norte)

## Models

| Model | Params | Architecture | Type |
|-------|--------|-------------|------|
| Naive (7d ago) | 0 | Same hour last week | Baseline |
| Chronos-2 | 120M | Encoder-only Transformer | Zero-shot |
| TiRex | 35M | xLSTM-based | Zero-shot |
| Moirai 2.0 | 11M | Decoder-only Transformer | Zero-shot |

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Download data

```bash
# Download 2019-2025 (default)
python scripts/download_ons.py

# Download more history
python scripts/download_ons.py --start 2000 --end 2025

# Single subsystem
python scripts/download_ons.py --subsystem SE
```

### 2. Run benchmark

```bash
# All models, SE subsystem, 24h forecast horizon
python scripts/benchmark.py

# Specific subsystem and models
python scripts/benchmark.py --subsystem NE --models naive chronos tirex

# 1-hour ahead prediction
python scripts/benchmark.py --horizon 1

# Longer context window (30 days)
python scripts/benchmark.py --context-length 720

# Force CPU
python scripts/benchmark.py --device cpu
```

Results saved to `results/`.

## Context for Claude Code

This project benchmarks zero-shot time series foundation models on Brazilian electricity load data from ONS. The key question is whether models pre-trained on diverse global time series can accurately forecast Brazilian load without any training on local data. The naive baseline (same hour from 7 days ago) captures weekly seasonality and is the bar to beat. Metrics: MAE (MW), RMSE (MW), MAPE (%), MASE, RMSSE, R2. Data is hourly, 4 subsystems. SE (Sudeste) is the largest and most important subsystem (~55% of national load).
