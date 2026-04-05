# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Benchmarking zero-shot time series foundation models on Brazilian electricity load forecasting using ONS data. The paper's thesis is "universality of electricity demand" — patterns are physics-driven and transfer across grids without local training. Paper targets Applied Energy or IEEE TPWRS.

## Key results (SE subsystem, 24h horizon)

| Model | Type | MAPE |
|-------|------|------|
| Chronos-2 (fine-tuned) | Fine-tuned | 1.73% |
| Chronos-2 | Zero-shot | 1.86% |
| Moirai 2.0 | Zero-shot | 1.93% |
| N-BEATS (tuned) | Trained 5+ years | 1.91% ± 0.08% |
| TiRex | Zero-shot | 2.33% |
| Naive (7d ago) | Baseline | 5.13% |

Data leakage cleared: ONS not in Chronos training corpus.

## Commands

```bash
# Download data
python scripts/download_ons.py --subsystem SE

# Benchmarks (all models, all subsystems)
python scripts/benchmark.py
python scripts/benchmark.py --subsystem NE --horizon 168
python scripts/benchmark.py --test-year 2023 --models naive chronos

# Trained baselines
python scripts/train_nbeats.py                    # N-BEATS via Darts (CPU only, MPS breaks)
python scripts/train_baselines.py                 # LSTM + Linear

# Fine-tuning
python scripts/finetune_chronos.py --device cpu   # MPS breaks for fine-tuning

# Analysis
python scripts/error_analysis.py                  # MAPE by hour/day/holiday
python scripts/probabilistic_eval.py              # CRPS, calibration, PI
python scripts/context_ablation.py                # Context length sweep
```

## Architecture

Scripts in `scripts/`, data in `data/`, results in `results/`. No library structure.

**Model API quirks:**
- Chronos-2: `BaseChronosPipeline`, input `(1, 1, seq_len)` 3D, returns `list[Tensor]` shape `(variates, quantiles, pred_len)`
- TiRex: `load_model('NX-AI/TiRex')`, input `(1, seq_len)` 2D, returns `(quantiles, mean)`
- Moirai 2.0: `Moirai2Forecast` + `Moirai2Module` (NOT v1 `MoiraiModule`), input `list[np.ndarray]`, returns `(batch, 9_quantiles, pred_len)`, index 4 = median
- **MPS breaks** for Darts N-BEATS (float64) and Chronos fine-tuning (fused AdamW). Use `--device cpu` or `accelerator: cpu`.

## RESOLVED: Reviewer issue W2

**N-BEATS tuning complete.** Grid search over 9 configs (input_length ∈ {168,336,720} × lr ∈ {1e-4,5e-4,1e-3}), best config (168h, lr=5e-4) run 3× with seeds {42,123,7}. Result: **1.91% ± 0.08% MAPE** (improved from 2.14%). Moirai (1.93%) vs tuned N-BEATS (1.91%) is now a statistical tie at comparable model size — but zero-shot requires no local training. Chronos-2 (1.86%) still wins outright. Script: `scripts/nbeats_sweep.py`.

## Other minor TODOs

- ~~M2: Replace approximate dataset stats (~61,000) with exact numbers~~ DONE (61,368 rows, exact means)
- ~~M5: Fill in [TODO] references in Related Work section~~ DONE (Santos et al. 2023, Velasquez et al. 2022, de Oliveira et al. 2023)
- Remove "Draft — Work in Progress" header when ready to submit

## Paper structure

`DRAFT_PAPER.md` — 10 figures, 7 experimental sections, universality framing. All experimental results are complete except W2 N-BEATS tuning.

## Active experimental direction (branch: holiday-covariates)

**Concern:** current paper reads like a Chronos ad — load forecasting is the easy case where FMs should obviously win. Need a novel angle.

**Hypotheses (falsification tests, in order):**
- **H1** — zero-shot FMs degrade on Brazilian holidays/Carnaval/bridge days vs normal weekdays. If FALSE → pivot to other OOD regimes (extreme weather, COVID, World Cup).
- **H2** — adding minimal holiday covariates (binary + days-until) recovers FM performance WITHOUT retraining weights.
- **H3** (strongest narrative) — covariate gain is LARGER for FMs than N-BEATS, because N-BEATS already learned BR holidays implicitly from 5+ yrs training while FM has not. Flips the story to "universality + cheap transfer beats trained-from-scratch."

**H1 pilot ready.** Run on mac-mini:
```bash
pip install holidays
python scripts/holiday_analysis.py --test-year 2024
# caches predictions under results/preds_*/ for fast iteration
```
Outputs conditional MAPE + lift-vs-normal table per model. Check `holiday` and `carnaval` columns for each model — if lift ≈ 1.0 everywhere, H1 is falsified.

H2 covariate experiment to be designed AFTER H1 results come back.
