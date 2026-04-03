# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Benchmarking zero-shot time series foundation models (Chronos-2, TiRex, Moirai 2.0) on Brazilian electricity load forecasting using ONS (Operador Nacional do Sistema Eletrico) hourly data. This is a research project targeting publication — the novel claim is that zero-shot foundation models achieve ISO-grade accuracy (1.86% MAPE) on Brazil's hydro-dependent grid without any local training.

## Commands

```bash
# Download ONS hourly load data (downloads to data/raw/, processes to data/processed/)
python scripts/download_ons.py --subsystem SE              # single subsystem
python scripts/download_ons.py --start 2000 --end 2025     # custom year range

# Run benchmark (outputs CSV + PNG to results/)
python scripts/benchmark.py                                 # all models, SE, 24h horizon
python scripts/benchmark.py --subsystem NE --horizon 168    # specific subsystem/horizon
python scripts/benchmark.py --models naive chronos --device cpu  # subset of models, force CPU
```

## Architecture

Two-script pipeline, no library/package structure:

1. **`scripts/download_ons.py`** — Fetches yearly CSVs from ONS S3 (`ons-aws-prod-opendata` bucket), renames columns to English, concatenates, and saves as parquet + CSV in `data/processed/`. Raw CSVs are semicolon-delimited with columns: `id_subsistema`, `nom_subsistema`, `din_instante`, `val_cargaenergiahomwmed`.

2. **`scripts/benchmark.py`** — Loads processed data, splits last N days as test (default 365), runs rolling forecasts in `horizon`-step increments using a 720-hour (30-day) context window. Each model runner function (`run_chronos`, `run_tirex`, `run_moirai`, `run_naive`) returns a numpy array of predictions. Metrics are computed via `evaluate()` using seasonality=24 for MASE/RMSSE.

**Model API differences to be aware of:**
- Chronos-2: `BaseChronosPipeline`, input shape `(1, 1, seq_len)` (3D), returns `list[Tensor]` of shape `(n_variates, n_quantiles, pred_len)`, median = middle quantile index
- TiRex: `load_model('NX-AI/TiRex')`, input shape `(1, seq_len)` (2D), returns `(quantiles, mean)`, use `mean` for point forecast
- Moirai 2.0: `Moirai2Forecast` + `Moirai2Module` (NOT `MoiraiModule` — that's v1), input is `list[np.ndarray]`, returns `(batch, 9_quantiles, pred_len)`, index 4 = median (0.5 quantile)

## Data

- **Source:** https://dados.ons.org.br/ (CC-BY 4.0, no auth required)
- **S3 pattern:** `https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/curva-carga-ho/CURVA_CARGA_{YEAR}.csv`
- **Subsystems:** SE (Sudeste, ~40k MW), S (Sul, ~14k MW), NE (Nordeste, ~13k MW), N (Norte, ~8k MW)
- `data/raw/` is gitignored; `data/processed/` is gitignored. User must run download_ons.py first.

## Research context

- **Paper draft:** `DRAFT_PAPER.md` — structured academic paper with results for all 4 subsystems and international comparison
- **Intuition guide:** `INTUITION.md` — plain English explanation of why this works
- **Key result:** Chronos-2 achieves 1.67-3.17% MAPE across all 4 subsystems (zero-shot), beating naive baselines by 45-64%
- **Comparison:** Matches PJM (US) proprietary forecasts at 1.78-1.98% MAPE
- **Related repo:** https://github.com/nelsonbarlow/xlstm-ts — prior work on xLSTM-TS stock prediction paper replication (concluded that stock directional prediction is not viable)

## Reviewer critique guidance

When suggesting improvements to the paper or experiments, adopt the perspective of a tier-1 venue reviewer (NeurIPS, ICML, AAAI). Key weaknesses to address:
- No locally-trained model comparison (LSTM, N-BEATS on ONS data)
- Univariate only (no weather/calendar exogenous features)
- Single test year
- No probabilistic evaluation (CRPS, calibration)
- No extreme event / holiday analysis
