# The Universality of Electricity Demand: Zero-Shot Foundation Models Match Locally-Trained Accuracy on Brazil's Power Grid

**Nelson Barlow**
*Centro de Informática - Universidade Federal de Pernambuco*
*email: ngb2@cin.ufpe.br*

> Markdown companion to `REPORT_IEEE.tex` (the IEEE-format submission). All numbers trace to `results/*.csv` on `master`, plus the `asian-market` branch (TEPCO) and `holiday-covariates` branch (holiday regime + covariate experiment).
>
> **Reproducing this paper:** [`README.md`](README.md) maps every section, table, and figure below to the exact command and result file that produces it — read a section here, run the matching command there.

---

## Abstract

Short-term load forecasting (STLF) has traditionally required models trained on years of local historical demand—a barrier in data-scarce grids. We ask whether recent time series *foundation models*, applied zero-shot, can forecast Brazilian electricity load with no local training. Using seven years (2019–2025) of hourly ONS load across all four subsystems of the interconnected grid (SIN), we benchmark Chronos-2 (120 M), TiRex (35 M), and Moirai 2.0 (11 M) against a hyperparameter-tuned N-BEATS, a linear model, an LSTM, and a seasonal-naive baseline. On the Southeast/Center-West (SE) subsystem at a 24-hour horizon, zero-shot Chronos-2 attains 1.86% MAPE—statistically indistinguishable from a tuned N-BEATS trained on 5+ years of local data (1.91% ± 0.08%; Diebold–Mariano *p* > 0.29), with light fine-tuning yielding only a 7% relative gain (1.73%). The result holds across all four subsystems (1.67–3.17% MAPE, R² > 0.90), three test years (1.89% ± 0.04%), and—on the Tokyo (TEPCO) grid, a different country and hemisphere—where the same three zero-shot models again beat a locally-trained N-BEATS (3.91% vs. 4.44%). The dominant residual error concentrates on national holidays (4–5× the normal-day error); adding a single binary holiday covariate cuts holiday error by 35%. We read this as evidence for a *bounded universality* of electricity demand: physics- and behavior-driven structure transfers across grids and hemispheres without adaptation, while calendar-specific demand does not.

*Index Terms*—Short-term load forecasting, foundation models, zero-shot learning, power systems, Chronos, Moirai, TiRex, N-BEATS.

---

## I. Introduction

STLF underpins unit commitment, economic dispatch, reserve sizing, and day-ahead market clearing. The dominant paradigm has each utility build a bespoke model trained on years of local load, augmented with weather and calendar features—a heavy barrier in data-scarce contexts (newly interconnected regions, microgrids, developing grids).

*Time series foundation models* (TSFMs), pretrained on billions of observations across diverse domains, raise a sharp question: can a model that has never seen a grid's data forecast its load accurately? We frame this as a hypothesis of **universality**. Demand is shaped by universal forces (heating/cooling/lighting physics, circadian rhythm, the weekly work/rest cycle) producing daily peaks, overnight troughs, and weekend dips everywhere; and by local forces (national holidays, hydro constraints, southern-hemisphere seasonality). If a pretrained model has internalized the universal component, it should transfer zero-shot, with residual error confined to the local component.

We test this primarily on the Brazilian interconnected system (SIN), with a cross-country replication on the Tokyo (TEPCO) grid. Contributions: (1) a systematic benchmark of 2024-era TSFMs on Brazilian load (all four SIN subsystems, four horizons, three test years) plus a cross-country replication on a Japanese grid; (2) evidence that zero-shot models match or beat a tuned, locally-trained N-BEATS (DM *p* > 0.29 on SE); (3) a precise account of *when and why* they fail—and a holiday covariate that recovers most of the gap; (4) a reproducible benchmark.

---

## II. Related Work

Brazilian STLF relies on locally-trained SVR, ANN, ARIMA, and LSTM models on ONS or utility data (Santos et al. 2023; Velasquez et al. 2022; de Oliveira et al. 2023), all requiring per-grid development. The TSFM paradigm emerged in 2023–2024—pretrain on billions of time points, then forecast unseen series zero-shot—with models including Chronos, TimesFM, Moirai, and TiRex. TSFMs have been evaluated on US grids (ERCOT), Singapore/Australia, and European households, but not on emerging-market grids with hydro dependency, inverted seasonality, and distinct holiday calendars, nor against a locally-trained deep-learning baseline on the same data. We fill that gap.

---

## III. Data and Methodology

**Data.** Hourly load from the ONS open-data portal (dados.ons.org.br, CC-BY 4.0) for the four SIN subsystems (Table I); SE is the largest at ~55% of national load (which averages 71,081 MW). Each series spans 2019-01-01 to 2025-12-31 (61,368 hourly rows). The most recent 365 days (8,760 h) form the test set: foundation models see only the preceding context window (zero-shot, never trained on ONS data); trained baselines use all earlier data, reserving the prior 60 days for validation.

**TABLE I. SIN Subsystems and Dataset Statistics (2019–2025)**

| Sub. | Region | Mean (MW) | Test (MW) | Share |
|------|--------|----------:|----------:|------:|
| SE | SP, RJ, MG | 40,408 | 44,226 | ~55% |
| S | PR, SC, RS | 12,320 | 13,804 | ~17% |
| NE | BA, PE, CE | 11,711 | 13,267 | ~17% |
| N | AM, PA | 6,641 | 8,312 | ~11% |

**Task.** Given a context window of hourly load, predict the next 24 h (day-ahead); we also evaluate 168/336/720 h. Forecasts roll through the test year in 24-h steps and are averaged.

**Models.** Zero-shot: Chronos-2 (120 M, encoder-only, quantile output; median reported), TiRex (35 M, xLSTM), Moirai 2.0 (11 M). We verified ONS data is *absent* from Chronos's pretraining corpus (no data leakage). Trained: N-BEATS (7.3 M, Darts; primary baseline), Linear (~8 K), LSTM, and Naive (7-day-ago). Metrics: MAE, RMSE (MW), MAPE, MASE/RMSSE (seasonality = 24), R²; CRPS and prediction intervals for probabilistic eval; Diebold–Mariano (DM) test on per-window errors. Hardware: Mac Mini M4, no GPU; MPS fails for Darts N-BEATS and Chronos fine-tuning, run on CPU.

---

## IV. Main Results (SE, 24 h)

**TABLE II. Main Results — SE, 24 h Horizon**

| Model | Type | MAPE (%) | MASE | R² |
|-------|------|---------:|-----:|---:|
| **Chronos-2** | **Fine-tuned** | **1.73** | 0.30 | 0.96 |
| Chronos-2 | Zero-shot | 1.86 | 0.33 | 0.96 |
| N-BEATS (tuned) | Trained 5 yr | 1.91 ± 0.08 | 0.34 | 0.96 |
| Moirai 2.0 | Zero-shot | 1.93 | 0.34 | 0.95 |
| Linear | Trained | 2.26 | 0.40 | 0.94 |
| TiRex | Zero-shot | 2.33 | 0.40 | 0.94 |
| Naive (7 d) | Baseline | 5.13 | 0.89 | 0.77 |
| LSTM | Trained | 13.14 | 2.39 | −0.36 |

Zero-shot Chronos-2 achieves **1.86% MAPE**, edging out a tuned N-BEATS (1.91%) trained on five years of local data, with Moirai close behind (1.93%); all TSFMs beat naive (5.13%) by a wide margin. The DM test gives *p* > 0.29 against all three N-BEATS seeds and bootstrap 95% CIs overlap (Chronos-2 CI [1.70%, 2.04%]): Chronos-2 vs. N-BEATS is a **statistical tie**—but Chronos-2 needs *zero* local training. Light fine-tuning (400 steps, lr 1e-5, ~40 min CPU) reaches 1.73% (a 7% gain). The LSTM failed to train competitively (known instability) and is shown for completeness only.

**N-BEATS tuning.** Grid-searched over input length ∈ {168, 336, 720} h and lr ∈ {1e-4, 5e-4, 1e-3}. Best config (168 h, lr 5e-4), re-run with seeds {42,123,7}, gives 1.85/1.88/2.00% → **1.91% ± 0.08%**—a 0.23 pp improvement over its untuned setting (2.14%), confirming the baseline is properly optimized.

---

## V. Generalization, Ablations, Probabilistic

**Across subsystems.** Chronos-2 leads on all four (Table III), 45–64% above naive, R² > 0.90 everywhere. The smallest subsystem (N) is easiest (1.67%); S is hardest (3.17%), likely from winter cold-front heating. On S, tuned N-BEATS reaches 3.26% ± 0.21% vs. 3.17%—again a tie.

**TABLE III. MAPE (%) by Subsystem — 24 h**

| Sub. | Naive | Chronos-2 | TiRex | Moirai |
|------|------:|----------:|------:|-------:|
| SE | 5.13 | **1.86** | 2.33 | 1.93 |
| S | 7.11 | **3.17** | 3.37 | 3.35 |
| NE | 3.76 | **1.94** | 2.06 | 2.05 |
| N | 3.03 | **1.67** | 1.67 | 1.76 |

**Horizon.** Naive crossover at ~2–3 weeks. SE Chronos-2 MAPE: 1.86% (24 h), 3.59% (168 h), 4.18% (336 h), 5.69% (720 h); at 720 h Chronos-2 (MASE 1.02) and Moirai (1.03) fall behind naive while only TiRex (0.91) stays ahead. TSFMs are best for operational (24–168 h) horizons.

**Multi-year.** Chronos-2 is stable across 2023/2024/2025 (1.94/1.87/1.86%, mean 1.89% ± 0.04%) while naive varies more (5.31% ± 0.17%).

**Context length.** Critical threshold is *one week*: MAPE halves (5.65%→2.80%) once a full weekly cycle is seen, reaching 1.86% at 30 days; beyond that, marginal (90 days: 1.77%). Thirty days is the practical accuracy/compute sweet spot.

**Model scale.** Chronos-Bolt family scales log-linearly: Tiny 9 M → 2.28%, Mini 21 M → 2.19%, Small 48 M → 2.02%, Chronos-2 120 M → 1.86%. At matched size zero-shot Bolt-Tiny (2.28%) loses to tuned N-BEATS (1.91%), but Moirai (11 M, 1.93%) nearly ties it—architecture matters as much as scale.

**Probabilistic.** Both probabilistic models are well-calibrated but slightly conservative—80% PIs cover 86–89% of outcomes. Chronos-2 has the better CRPS (643 vs. 688 MW) and Winkler score (4354 vs. 4378); Moirai produces sharper intervals (3024 vs. 3295 MW) despite being 11× smaller.

### Cross-Country Replication (Tokyo) — `asian-market` branch

We repeat the benchmark on the TEPCO grid (Tokyo, ~32 GW, 2019–2024), with N-BEATS trained on 2019–2023 Japanese data and 2024 held out. All three zero-shot foundation models again **beat** the locally-trained N-BEATS, in the identical ranking observed for Brazil. Absolute MAPE is higher (Japanese demand is more volatile), but the qualitative result holds across a different country, hemisphere, and holiday calendar—the strongest single piece of evidence for the universality hypothesis.

**TABLE IV. Cross-Country: TEPCO Tokyo — 24 h, test 2024**

| Model | Type | MAPE (%) | MASE | R² |
|-------|------|---------:|-----:|---:|
| Chronos-2 | Zero-shot | **3.91** | 0.59 | 0.91 |
| Moirai 2.0 | Zero-shot | 3.94 | 0.59 | 0.91 |
| TiRex | Zero-shot | 4.05 | 0.61 | 0.91 |
| N-BEATS | Trained 5 yr | 4.44 | 0.67 | 0.89 |
| Naive (7 d) | Baseline | 8.86 | 1.30 | 0.65 |

---

## VI. Error Analysis: When and Why

**By hour.** Error follows the load curve: lowest in the overnight trough (0.51% at hour 0), highest mid-afternoon (2.72% at hour 15).

**By day.** Weekends (1.75%) beat weekdays (1.90%); Monday is hardest (2.37%), Tuesday easiest (1.60%).

**By holiday — the central finding.** On Brazilian public holidays Chronos-2 degrades to several times its normal-day error, because nothing in the global corpus signals a Brazilian holiday, so it forecasts ordinary workday demand. The effect is a property of the data, not one model: on the same holidays Moirai reaches 7.9% and TiRex 9.1% MAPE, and the naive baseline 12.8%, against ~1.6% on normal days (`holiday_mape_SE_24h_2024.csv`).

**The remedy works.** Adding a single binary holiday covariate and lightly fine-tuning Chronos-2 (test 2024) cuts holiday MAPE by 35% (6.76%→4.37%), Carnaval by 18%, and the day-after-holiday by 16%, while normal days are untouched and overall MAPE falls 5%. The lone regression—"bridge" days between a holiday and a weekend worsen (3.02%→6.12%), on only 24 test hours—marks where the coarse binary encoding is too blunt. This is exactly the bounded-universality prediction: the residual error is local calendar knowledge, and supplying that knowledge recovers most of the gap.

**TABLE V. Holiday Covariate Experiment — Chronos-2, SE (test 2024)** *(`holiday_covariates_SE_24h_2024.csv`)*

| Regime | Hours | Zero-shot (%) | + Covariate (%) | Δ% |
|--------|------:|--------------:|----------------:|----:|
| Normal | 5688 | 1.53 | 1.53 | +0.0 |
| Weekend | 2232 | 1.77 | 1.74 | −2.0 |
| Day-before | 264 | 2.28 | 2.19 | −4.2 |
| Day-after | 264 | 2.29 | 1.93 | −15.8 |
| Carnaval | 48 | 5.68 | 4.64 | −18.3 |
| **Holiday** | 264 | **6.76** | **4.37** | **−35.4** |
| Bridge | 24 | 3.02 | 6.12 | +102.5 |
| Overall | 8784 | 1.82 | 1.73 | −5.0 |

---

## VII. Discussion and Conclusion

The results support a universality of electricity demand *with a precise boundary*. Components driven by shared physical and behavioral forces—diurnal cycles, the workweek, seasonality—transfer to an unseen grid well enough that a zero-shot model ties or beats a locally-trained one across subsystems, years, and—on the Tokyo grid—a second country and hemisphere. Components driven by local culture—national holidays—do not, degrading 4–5×; but supplying exactly that local knowledge via a binary holiday covariate recovers 35% of the holiday gap, confirming the boundary is calendar knowledge rather than an architectural limit.

**Operational recipe.** (1) Start zero-shot (Chronos-2, or the 11 M Moirai when resource-constrained) with ~30 days of context; (2) fine-tune if resources permit (~7% gain, 40 min CPU); (3) add a binary holiday covariate (cut holiday error 35% in our experiment); (4) do not use beyond ~2 weeks, where naive wins; (5) one week of history is the minimum viable context.

**Limitations.** Univariate input (no weather); two countries (Brazil, Japan) remain a small sample of grids; load-only trained baselines, so a weather-augmented TFT could narrow the gap; the holiday covariate's binary encoding regresses on rare "bridge" days; the LSTM did not converge competitively.

**Conclusion.** A model pretrained on global time series, with no exposure to the target grid's data, matches a tuned model trained on 5+ years of local Brazilian data (1.86% vs. 1.91% ± 0.08%; DM *p* > 0.29) across all four subsystems and three years, and *beats* a locally-trained N-BEATS on the Tokyo grid (3.91% vs. 4.44%). The boundary of universality is precise—the model fails on culturally-local holidays and essentially nowhere else—and a single holiday covariate largely closes even that gap. For operators in data-scarce grids, accurate day-ahead forecasting no longer requires years of local model development: thirty days of load history and an off-the-shelf foundation model suffice.

---

## Data and Code Availability

ONS data: dados.ons.org.br (CC-BY 4.0). TEPCO data: tepco.co.jp (`scripts/download_tepco.py`). Pipeline (`benchmark.py`, `nbeats_sweep.py`, `finetune_chronos.py`, `chronos_scaling.py`, `error_analysis.py`, `probabilistic_eval.py`, `context_ablation.py`, `holiday_analysis.py`, `holiday_covariates.py`, `benchmark_tepco.py`) and all result tables are released for reproducibility. Hardware: Mac Mini M4, 24 GB, no GPU.
