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
| N-BEATS | Trained 5+ years | 2.14% |
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

## PRIORITY: Remaining reviewer issue W2

**N-BEATS may be under-tuned.** It early-stopped at epoch 30/200 with only one hyperparameter config. To address:

1. Try different input lengths: `--input-length 336` and `--input-length 720`
2. Try different learning rates: `--lr 5e-4` and `--lr 1e-3`
3. Run best config 3x with different seeds (add `--random-state` flag to script if needed)
4. Report mean +/- std

Even if N-BEATS improves, the key comparison is Moirai (11M, zero-shot) vs N-BEATS (7.3M, trained) at comparable model size. If Moirai still wins, the finding is robust.

## Other minor TODOs

- M2: Replace approximate dataset stats (~61,000) with exact numbers
- M5: Fill in [TODO] references in Related Work section
- Remove "Draft — Work in Progress" header when ready to submit

## Paper structure

`DRAFT_PAPER.md` — 10 figures, 7 experimental sections, universality framing. All experimental results are complete except W2 N-BEATS tuning.
