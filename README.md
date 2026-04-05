# Brazilian Energy Load Forecasting with Foundation Models

Benchmarking zero-shot time series foundation models on Brazilian electricity load forecasting using ONS (Operador Nacional do Sistema Elétrico) data. The thesis: electricity demand patterns are physics-driven and transfer across grids without local training.

Paper targets Applied Energy or IEEE TPWRS.

## Research Gap

No published paper with code applies modern foundation models (Chronos-2, TiRex, Moirai 2.0) to Brazilian energy data. This project fills that gap with a comprehensive benchmark across all four ONS subsystems.

## Why This Matters

- Brazil's grid is hydro-dependent (~65% of generation), creating unique seasonal patterns (dry/wet seasons) that differ from US/EU grids
- Foundation models pre-trained on diverse global time series may transfer zero-shot to Brazil's unique grid dynamics
- If zero-shot models match or beat locally trained baselines, it validates the "universality of electricity demand" hypothesis

## Key Results (SE subsystem, 24h horizon)

| Model | Type | MAPE |
|-------|------|------|
| Chronos-2 (fine-tuned) | Fine-tuned | 1.73% |
| Chronos-2 | Zero-shot | 1.86% |
| N-BEATS (tuned) | Trained 5+ years | 1.91% ± 0.08% |
| Moirai 2.0 | Zero-shot | 1.93% |
| TiRex | Zero-shot | 2.33% |
| Naive (7d ago) | Baseline | 5.13% |

Zero-shot Chronos-2 beats a tuned N-BEATS trained on 5+ years of local data. Data leakage cleared: ONS is not in the Chronos training corpus.

## Data

**Source:** [ONS Open Data Portal](https://dados.ons.org.br/)

- **License:** CC-BY 4.0 (free, open access, no registration)
- **Resolution:** Hourly (61,368 rows for SE subsystem)
- **History:** 2000-2026 (26 years)
- **Subsystems:** SE (Sudeste/Centro-Oeste), S (Sul), NE (Nordeste), N (Norte)

## Models

| Model | Params | Architecture | Type |
|-------|--------|-------------|------|
| Naive (7d ago) | 0 | Same hour last week | Baseline |
| N-BEATS | ~6M | Fully connected blocks | Trained |
| LSTM | ~1M | Recurrent | Trained |
| Linear | ~50K | Linear regression | Trained |
| Chronos-2 | 8M-710M | Encoder-only Transformer | Zero-shot |
| TiRex | 35M | xLSTM-based | Zero-shot |
| Moirai 2.0 | 11M | Decoder-only Transformer | Zero-shot |
| Chronos-2 (fine-tuned) | 120M | Encoder-only Transformer | Fine-tuned |

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Download data

```bash
python scripts/download_ons.py                     # Download 2019-2025 (default)
python scripts/download_ons.py --start 2000        # Download more history
python scripts/download_ons.py --subsystem SE      # Single subsystem
```

### 2. Run benchmarks

```bash
python scripts/benchmark.py                                          # All models, SE, 24h
python scripts/benchmark.py --subsystem NE --horizon 168             # NE, 1-week horizon
python scripts/benchmark.py --test-year 2023 --models naive chronos  # Specific models
python scripts/benchmark.py --device cpu                             # Force CPU
```

### 3. Trained baselines

```bash
python scripts/train_nbeats.py                     # N-BEATS via Darts
python scripts/train_baselines.py                  # LSTM + Linear
python scripts/nbeats_sweep.py                     # N-BEATS hyperparameter grid search
```

### 4. Fine-tuning

```bash
python scripts/finetune_chronos.py --device cpu    # Chronos-2 fine-tuning
```

### 5. Analysis

```bash
python scripts/error_analysis.py                   # MAPE by hour/day/holiday
python scripts/probabilistic_eval.py               # CRPS, calibration, prediction intervals
python scripts/context_ablation.py                 # Context length sweep
python scripts/chronos_scaling.py                  # Model scaling analysis (Bolt-Tiny → Chronos-2)
python scripts/statistical_comparison.py           # Statistical significance tests
```

Results saved to `results/`.

## Notes

- **MPS breaks** for Darts N-BEATS (float64) and Chronos fine-tuning (fused AdamW). Use `--device cpu` on Apple Silicon.
- SE (Sudeste) is the largest and most important subsystem (~55% of national load).
