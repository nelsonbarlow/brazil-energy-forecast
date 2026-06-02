# Brazilian Energy Load Forecasting with Foundation Models

Benchmarking zero-shot time series foundation models (Chronos-2, TiRex, Moirai 2.0) on
Brazilian electricity load forecasting using ONS (Operador Nacional do Sistema Elétrico) data.

**Thesis — universality of electricity demand:** load is shaped by *universal* forces
(heating/cooling physics, circadian rhythm, the weekly work/rest cycle) that transfer zero-shot
across grids, and by *local* forces (national holidays, hydro constraints) that do not. A model
pretrained on global time series, never trained on Brazilian data, matches a tuned N-BEATS trained
on 5+ years of local load — and beats a locally-trained N-BEATS on the Tokyo (TEPCO) grid.

## The paper and this repo, side by side

The full write-up lives in **[`REPORT_IEEE.md`](REPORT_IEEE.md)** (Markdown companion) and
**[`REPORT_IEEE.pdf`](REPORT_IEEE.pdf)** (IEEE-format submission, built from `REPORT_IEEE.tex`).

This README is organized to **track the report section by section**: every claim, table, and figure
in the paper maps to one command and one result file here. Read a section of the paper, then run the
matching command below to regenerate its numbers. Every number in the paper traces to a `results/*.csv`.

| Read in paper | Run here | Produces |
|---------------|----------|----------|
| §III Data, Table I | `python scripts/download_ons.py` | `data/` (61,368 hourly rows × 4 subsystems) |
| §IV Table II — main results (SE, 24 h) | `python scripts/benchmark.py` | `results/benchmark_SE_24h.csv` |
| §IV N-BEATS tuning (W2) | `python scripts/nbeats_sweep.py` | `results/nbeats_sweep_SE_24h.csv`, `nbeats_final_SE_24h.json` |
| §IV trained baselines (Linear, LSTM) | `python scripts/train_baselines.py` | `results/trained_vs_zeroshot_SE_24h.csv` |
| §IV N-BEATS as Darts baseline | `python scripts/train_nbeats.py` | `results/nbeats_comparison_SE_24h.csv` |
| §IV fine-tuning Chronos-2 | `python scripts/finetune_chronos.py --device cpu` | `results/finetune_SE_24h.csv` |
| §IV Diebold–Mariano tie (p > 0.29) | `python scripts/statistical_comparison.py` | `results/statistical_comparison_SE_24h.json` |
| §V Table III — across subsystems | `python scripts/benchmark.py --subsystem {S,NE,N}` | `results/benchmark_{S,NE,N}_24h.csv` |
| §V horizon sweep (168/336/720 h) | `python scripts/benchmark.py --horizon {168,336,720}` | `results/benchmark_SE_{168,336,720}h.csv` |
| §V multi-year (2023/24/25) | `python scripts/benchmark.py --test-year {2023,2024,2025}` | `results/benchmark_SE_24h_{2023,2024,2025}.csv` |
| §V context-length ablation | `python scripts/context_ablation.py` | `results/context_ablation_SE.csv` |
| §V model-scale (Bolt-Tiny → Chronos-2) | `python scripts/chronos_scaling.py` | `results/chronos_scaling_SE_24h.csv` |
| §V probabilistic (CRPS, PIs) | `python scripts/probabilistic_eval.py` | `results/probabilistic_SE_24h.csv` |
| §V Table IV — cross-country (TEPCO) | `python scripts/download_tepco.py && python scripts/benchmark_tepco.py` | `results/benchmark_TEPCO_24h_2024.csv` |
| §V TEPCO trained N-BEATS | `python scripts/train_nbeats_tepco.py` | `results/nbeats_tepco_24h_2024.csv` |
| §VI error by hour / day | `python scripts/error_analysis.py` | `results/error_analysis_SE.csv` |
| §VI holiday degradation (H1) | `python scripts/holiday_analysis.py` | `results/holiday_mape_SE_24h_2024.csv` |
| §VI Table V — holiday covariate (H2) | `python scripts/holiday_covariates.py` | `results/holiday_covariates_SE_24h_2024.csv` |
| §VI holiday control on N-BEATS (H3) | `python scripts/h3_nbeats_covariates.py` | `results/h3_nbeats_covariates_SE_24h.csv` |

> **Branch note:** all scripts and result CSVs above are on `master`. The TEPCO and holiday work was
> developed on the `asian-market` and `holiday-covariates` branches respectively and merged here.

## Headline result (SE subsystem, 24 h horizon — Table II)

| Model | Type | MAPE |
|-------|------|------|
| Chronos-2 (fine-tuned) | Fine-tuned | 1.73% |
| **Chronos-2** | **Zero-shot** | **1.86%** |
| Moirai 2.0 | Zero-shot | 1.93% |
| N-BEATS (tuned) | Trained 5+ years | 1.91% ± 0.08% |
| TiRex | Zero-shot | 2.33% |
| Naive (7 d ago) | Baseline | 5.13% |

Zero-shot Chronos-2 (1.86%) is a statistical tie with a tuned N-BEATS trained on five years of local
data (1.91%, Diebold–Mariano *p* > 0.29) — but needs **zero** local training. We verified ONS data is
absent from Chronos's pretraining corpus (no data leakage).

## Data

**Source:** ONS Open Data Portal (https://dados.ons.org.br/) — CC-BY 4.0, hourly, 2000–2026.
Four subsystems: SE (Sudeste/Centro-Oeste, ~55% of national load), S (Sul), NE (Nordeste), N (Norte).
Cross-country: TEPCO (Tokyo) via `scripts/download_tepco.py`.

## Models

| Model | Params | Architecture | Type |
|-------|--------|-------------|------|
| Naive (7 d ago) | 0 | Same hour last week | Baseline |
| Chronos-2 | 120 M | Encoder-only Transformer | Zero-shot |
| TiRex | 35 M | xLSTM-based | Zero-shot |
| Moirai 2.0 | 11 M | Decoder-only Transformer | Zero-shot |
| N-BEATS (tuned) | 7.3 M | Doubly-residual MLP (Darts) | Trained |
| Linear / LSTM | ~8 K / — | — | Trained |

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Hardware note:** developed on a Mac Mini M4 (no GPU). MPS fails for Darts N-BEATS (float64) and
Chronos fine-tuning (fused AdamW) — pass `--device cpu` / `accelerator: cpu` for those.

## Reproduce everything

```bash
python scripts/download_ons.py            # §III data
python scripts/benchmark.py               # §IV main table
python scripts/nbeats_sweep.py            # §IV N-BEATS tuning
python scripts/statistical_comparison.py  # §IV DM test
# ...then walk the table above for §V and §VI.
```

Results are written to `results/` as CSV (numbers) and PNG (figures).
