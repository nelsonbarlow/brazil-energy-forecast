# Zero-Shot Foundation Models for Short-Term Load Forecasting in the Brazilian Power Grid

**Draft — Work in Progress**

---

## Abstract

Short-term load forecasting (STLF) is critical for power system operation, particularly in Brazil's hydro-dependent grid where supply uncertainty amplifies the need for accurate demand prediction. We evaluate three state-of-the-art time series foundation models — Chronos-2 (Amazon, 120M parameters), TiRex (NX-AI, 35M), and Moirai 2.0 (Salesforce, 11M) — on day-ahead hourly load forecasting across Brazil's four electrical subsystems using data from ONS (Operador Nacional do Sistema Eletrico). Without any training on Brazilian data, Chronos-2 achieves 1.86% MAPE on the SE (Sudeste) subsystem, matching the accuracy of proprietary ISO forecasting systems in the US (PJM: 1.78-1.98%) and outperforming trained deep learning models on comparable European grids. To our knowledge, this is the first evaluation of time series foundation models on Brazilian electricity load data. Our results suggest that pre-trained foundation models can provide ISO-grade forecasting accuracy for emerging market grids without domain-specific training.

---

## 1. Introduction

### 1.1 Motivation

Accurate short-term load forecasting (STLF) underpins power system operations including unit commitment, economic dispatch, reserve scheduling, and energy trading. The Brazilian Interconnected Power System (SIN), operated by ONS, serves over 200 million people across four subsystems with distinct climatic and economic characteristics. Brazil's heavy reliance on hydropower (~65% of installed capacity) makes demand forecasting especially critical: when reservoirs are constrained, the cost of forecast errors escalates rapidly as expensive thermal generation must compensate.

Traditionally, each grid operator develops bespoke forecasting models trained on local historical data, incorporating regional weather, calendar effects, and economic indicators. This requires significant expertise, data infrastructure, and ongoing model maintenance. Recent advances in time series foundation models (TSFMs) — large models pre-trained on diverse global time series corpora — raise a compelling question: can these models provide accurate load forecasts for grids they have never seen?

### 1.2 Contributions

1. We present the **first evaluation of time series foundation models on Brazilian electricity load data**, benchmarking Chronos-2, TiRex, and Moirai 2.0 in a zero-shot setting.
2. We demonstrate that **zero-shot foundation models match or exceed the accuracy of trained models and proprietary ISO systems** on day-ahead forecasting, achieving 1.86% MAPE on the SE subsystem — comparable to PJM's proprietary system (1.78-1.98%).
3. We evaluate **cross-regional transfer across all four Brazilian subsystems** (SE, S, NE, N), showing consistent 45-64% improvement over naive baselines regardless of subsystem size, climate zone, or load profile (R² > 0.90 in all cases).
4. We analyze **forecast horizon sensitivity**, testing how accuracy degrades from 24h to 720h (1 month) horizons.
5. We provide an **open-source benchmark** with reproducible code and publicly available data from ONS.

### 1.3 Related Work

**Load forecasting in Brazil.** [TODO: survey existing Brazilian STLF papers — most use traditional ML (SVR, ANN, ARIMA) or LSTM models trained on local data. Cite Conte et al. on PLD prediction, any ONS-specific papers.]

**Time series foundation models.** The TSFM paradigm emerged in 2023-2024, analogous to large language models for text. Key models include Chronos (Ansari et al., 2024), TimesFM (Das et al., 2024), Moirai (Woo et al., 2024), and TiRex (NX-AI, 2025). These models are pre-trained on hundreds of billions of time points from diverse domains and can produce forecasts for unseen time series without task-specific training ("zero-shot").

**Foundation models for energy.** Recent work has applied TSFMs to electricity forecasting in the US (ERCOT: [cite arxiv 2602.10848]), Singapore and Australia ([cite arxiv 2602.05390]), and European household-level data ([cite arxiv 2410.09487]). However, no prior work evaluates TSFMs on Brazilian grid data, which presents unique characteristics due to hydro dependency, southern hemisphere seasonality, and emerging market demand growth patterns.

---

## 2. Data

### 2.1 Source

We use hourly load data from ONS's open data portal (https://dados.ons.org.br/), released under CC-BY 4.0. The dataset ("Curva de Carga Horaria") provides hourly average load in MW for each of Brazil's four electrical subsystems.

### 2.2 Subsystems

| Subsystem | Code | Region | Share of National Load | Characteristics |
|-----------|------|--------|----------------------|-----------------|
| Sudeste/Centro-Oeste | SE | Sao Paulo, Rio, Minas Gerais | ~55% | Largest industrial base, highest absolute load |
| Sul | S | Parana, Santa Catarina, Rio Grande do Sul | ~17% | Colder winters, distinct seasonal profile |
| Nordeste | NE | Bahia, Pernambuco, Ceara | ~17% | Hot climate, growing wind/solar generation |
| Norte | N | Amazonas, Para | ~11% | Smallest subsystem, tropical climate, isolated loads |

### 2.3 Dataset Statistics

| Subsystem | Period | Rows | Mean Load (MW) | Test Mean (MW) |
|-----------|--------|------|----------------|----------------|
| SE (Sudeste) | 2019-2025 | ~61,000 | ~40,000 | ~40,000 |
| S (Sul) | 2019-2025 | ~61,000 | ~14,000 | ~13,804 |
| NE (Nordeste) | 2019-2025 | ~61,000 | ~13,000 | ~13,000 |
| N (Norte) | 2019-2025 | ~61,000 | ~8,000 | ~8,000 |

Each subsystem contains approximately 61,000 hourly observations (7 years x 8,760 hours/year). Total national load across all four subsystems averages approximately 75,000 MW.

### 2.4 Train/Test Split

We use the most recent 365 days (8,760 hours) as the test set. All preceding data forms the context pool. No training is performed — foundation models are evaluated purely zero-shot.

---

## 3. Methodology

### 3.1 Task Definition

Given a context window of H historical hourly load values for a single subsystem, predict the next 24 hourly load values (day-ahead forecast). This is the standard operational horizon for day-ahead market clearing and unit commitment.

### 3.2 Models

**Chronos-2** (Amazon, 120M parameters). Encoder-only transformer pre-trained on 100B+ time points. Uses group attention and provides quantile forecasts. We report the median (0.5 quantile).

**TiRex** (NX-AI, 35M parameters). xLSTM-based architecture (extended Long Short-Term Memory). Published at NeurIPS 2025. Notable for achieving state-of-the-art results with far fewer parameters than transformer alternatives.

**Moirai 2.0** (Salesforce, 11M parameters). Decoder-only transformer. The smallest model in our evaluation at 11M parameters — 96% smaller than Chronos-2.

**Naive baseline** (same hour, 7 days ago). For each forecast hour, predict the load at the same hour exactly one week prior. This captures weekly seasonality and is a standard baseline in load forecasting literature.

### 3.3 Evaluation Protocol

- **Rolling forecast**: We step through the test set in 24-hour increments, producing a fresh 24-hour forecast at each step.
- **Context length**: 720 hours (30 days) of historical load.
- **Zero-shot**: No model parameters are updated on the ONS data.
- **Metrics**: MAE (MW), RMSE (MW), MAPE (%), MASE (seasonality=24), RMSSE (seasonality=24), R².

### 3.4 Infrastructure

All experiments run on a Mac Mini M4 with 24GB unified memory. No GPU required — all models fit comfortably on CPU/MPS.

---

## 4. Results

### 4.1 SE Subsystem (Sudeste)

| Model | MAE (MW) | RMSE (MW) | MAPE | MASE | RMSSE | R² |
|-------|----------|-----------|------|------|-------|-----|
| Naive (7d ago) | 2,264 | 3,027 | 5.13% | 0.89 | 0.81 | 0.77 |
| Chronos-2 | **829** | **1,318** | **1.86%** | **0.33** | **0.35** | **0.96** |
| Moirai 2.0 | 858 | 1,338 | 1.93% | 0.34 | 0.36 | 0.95 |
| TiRex | 1,018 | 1,589 | 2.33% | 0.40 | 0.42 | 0.94 |

### 4.2 Cross-Regional Results (All Subsystems, 24h Horizon)

All four subsystems show consistent improvement over the naive baseline.

**MAPE (%) by subsystem and model:**

| Subsystem | Mean Load (MW) | Naive | Chronos-2 | TiRex | Moirai 2.0 | Improvement vs Naive |
|-----------|---------------|-------|-----------|-------|------------|---------------------|
| SE (Sudeste) | ~40,000 | 5.13% | **1.86%** | 2.33% | 1.93% | 64% |
| S (Sul) | ~14,000 | 7.11% | **3.17%** | 3.37% | 3.35% | 55% |
| NE (Nordeste) | ~13,000 | 3.76% | **1.94%** | 2.06% | 2.05% | 48% |
| N (Norte) | ~8,000 | 3.03% | **1.67%** | **1.67%** | 1.76% | 45% |

**Full metrics for each subsystem:**

**S (Sul):**

| Model | MAE (MW) | RMSE (MW) | MAPE | MASE | RMSSE | R² |
|-------|----------|-----------|------|------|-------|-----|
| Naive (7d ago) | 972 | 1,330 | 7.11% | 0.78 | 0.73 | 0.77 |
| Chronos-2 | **437** | **663** | **3.17%** | **0.35** | **0.36** | **0.94** |
| TiRex | 461 | 716 | 3.37% | 0.37 | 0.39 | 0.93 |
| Moirai 2.0 | 460 | 690 | 3.35% | 0.37 | 0.38 | 0.94 |

**NE (Nordeste):**

| Model | MAE (MW) | RMSE (MW) | MAPE | MASE | RMSSE | R² |
|-------|----------|-----------|------|------|-------|-----|
| Naive (7d ago) | 495 | 698 | 3.76% | 0.80 | 0.75 | 0.70 |
| Chronos-2 | **254** | **382** | **1.94%** | **0.41** | **0.41** | **0.91** |
| TiRex | 267 | 405 | 2.06% | 0.43 | 0.43 | 0.90 |
| Moirai 2.0 | 268 | 401 | 2.05% | 0.44 | 0.43 | 0.90 |

**N (Norte):**

| Model | MAE (MW) | RMSE (MW) | MAPE | MASE | RMSSE | R² |
|-------|----------|-----------|------|------|-------|-----|
| Naive (7d ago) | 251 | 340 | 3.03% | 0.83 | 0.78 | 0.76 |
| Chronos-2 | **138** | **199** | 1.67% | **0.46** | **0.45** | **0.92** |
| TiRex | **138** | **197** | **1.67%** | **0.46** | **0.45** | **0.92** |
| Moirai 2.0 | 145 | 206 | 1.76% | 0.48 | 0.47 | 0.91 |

### 4.3 Forecast Horizon Sensitivity (SE Subsystem)

We evaluate how accuracy degrades as the forecast horizon extends from 24 hours to 30 days.

**MAPE (%) by horizon:**

| Horizon | Naive | Chronos-2 | TiRex | Moirai 2.0 | Best vs Naive |
|---------|-------|-----------|-------|------------|---------------|
| 24h (1 day) | 5.13% | **1.86%** | 2.33% | 1.93% | 64% better |
| 168h (1 week) | 5.13% | **3.59%** | 3.74% | 3.69% | 30% better |
| 336h (2 weeks) | 5.13% | **4.18%** | 4.23% | 4.41% | 19% better |
| 720h (1 month) | **5.13%** | 5.69% | 5.17% | 5.76% | Naive wins |

**R² by horizon:**

| Horizon | Naive | Chronos-2 | TiRex | Moirai 2.0 |
|---------|-------|-----------|-------|------------|
| 24h | 0.77 | **0.96** | 0.94 | 0.95 |
| 168h | 0.77 | **0.87** | 0.87 | 0.87 |
| 336h | 0.77 | **0.83** | 0.82 | 0.81 |
| 720h | **0.77** | 0.66 | 0.76 | 0.67 |

**MASE by horizon (values > 1.0 indicate worse than naive):**

| Horizon | Chronos-2 | TiRex | Moirai 2.0 |
|---------|-----------|-------|------------|
| 24h | 0.33 | 0.40 | 0.34 |
| 168h | 0.62 | 0.65 | 0.64 |
| 336h | 0.73 | 0.74 | 0.77 |
| 720h | **1.02** | 0.91 | **1.03** |

Foundation models dominate at operational horizons (24h-168h) but degrade past the naive crossover point at approximately 2-3 weeks. At 720h (1 month), Chronos-2 and Moirai 2.0 both exceed MASE 1.0, indicating they are formally worse than the naive weekly-repetition baseline. TiRex (MASE 0.91) remains marginally better than naive at this extreme horizon, suggesting that the xLSTM architecture's state-tracking capability may provide an advantage for very long-range forecasting.

### 4.3 Comparison with International Benchmarks

| Benchmark | Region | MAPE | Method | Our Chronos-2 |
|-----------|--------|------|--------|---------------|
| PJM official 24h | US | 1.78-1.98% | Proprietary | 1.86% |
| ERCOT official | US | 1.66-3.73% | Proprietary | 1.86% |
| N-BEATS (trained) | Portugal | 1.90% | Trained DL | 1.86% |
| ANN multi-country | Europe (4) | 2.80% | Trained NN | 1.86% |
| Chronos-2 zero-shot | Singapore | ~1-2% | Zero-shot | 1.86% |
| Chronos-2 zero-shot | Australia | ~2-4% | Zero-shot | 1.86% |

### 4.4 Analysis

[TODO: Expand each of these]

**Model ranking.** Chronos-2 > Moirai 2.0 > TiRex > Naive across all metrics on SE. The 120M-parameter Chronos-2 leads, but the 11M-parameter Moirai 2.0 is remarkably close (1.93% vs 1.86%), suggesting diminishing returns from model scale for this task.

**Zero-shot vs trained models.** Our zero-shot results match or beat trained models reported in the literature for comparable grid sizes and horizons. This is notable because no hyperparameter tuning, feature engineering, or local data preparation was required.

**Naive baseline strength.** The naive baseline (MAPE 5.13%) is not trivial — it captures the strong weekly seasonality in electricity demand. Foundation models must learn to do better than this, which they clearly do (63% improvement for Chronos-2).

**Cross-subsystem variation.** Foundation models beat the naive baseline on all four subsystems (45-64% MAPE reduction), demonstrating robust transfer across regions with distinct characteristics. Contrary to our initial hypothesis, the smallest subsystem (Norte, ~8,000 MW) achieved the *best* MAPE (1.67%), not the worst. Sul (southern Brazil) was the hardest to forecast (3.17% MAPE), likely due to its more variable climate with cold fronts in winter creating demand spikes for heating. The naive baseline itself varied substantially: Sul had the weakest naive (7.11%) while Norte had the strongest (3.03%), suggesting that demand regularity varies more across subsystems than forecasting difficulty for foundation models.

**Model ranking consistency.** Chronos-2 (120M params) wins or ties on every subsystem. However, the gap between models is small: on Norte, TiRex (35M params) matches Chronos-2 exactly. Moirai 2.0 (11M params) is consistently within 0.1-0.2% MAPE of Chronos-2, suggesting diminishing returns from model scale. For resource-constrained deployment, the 11M-parameter Moirai may offer the best accuracy-per-parameter tradeoff.

**R² > 0.90 everywhere (at 24h).** All foundation models explain over 90% of load variance across all four subsystems at the 24-hour horizon, confirming that zero-shot transfer is robust and not dependent on subsystem size or geographic characteristics.

**Horizon decay and the naive crossover.** Foundation model accuracy degrades predictably with horizon length (Figure X). At 24h, Chronos-2 achieves 64% lower MAPE than naive; by 168h (1 week) this advantage shrinks to 30%; and at 720h (1 month) the naive baseline wins outright. This crossover occurs because the naive baseline's core assumption — that demand repeats weekly — becomes increasingly accurate at longer horizons where daily noise averages out. Foundation models, generating autoregressively, accumulate error with each step. The practical implication is clear: foundation models are most valuable for operational horizons (day-ahead to week-ahead), while simple seasonal baselines suffice for monthly planning. Notably, TiRex maintains MASE < 1.0 even at 720h, suggesting xLSTM's recurrent state-tracking may be better suited than transformer architectures for very long-range energy forecasting.

---

## 5. Discussion

### 5.1 Implications for Grid Operators

Our findings suggest that foundation models can serve as strong baseline forecasters for grid operators in emerging markets, potentially reducing the time and expertise required to deploy accurate STLF systems. A grid operator could achieve ISO-grade accuracy using an off-the-shelf model with no local training — only requiring historical load data as input context.

### 5.2 Limitations

1. **Univariate input only.** We use only historical load as input. Operational forecasting systems incorporate weather forecasts, calendar features, economic indicators, and planned outages. Adding exogenous variables would likely improve results further.
2. **No extreme event analysis.** We do not separately analyze performance during holidays, extreme weather events, or the 2021 water crisis. Foundation models may struggle with distributional shifts not well-represented in their pre-training data.
3. **Test period.** Our test set covers a single year (the most recent 365 days). Results may vary across years with different economic conditions or weather patterns.
4. **No comparison with locally-trained models.** We compare only against a naive baseline and international benchmarks. A direct comparison with models trained on ONS data (e.g., LSTM, N-BEATS) would strengthen the findings.

### 5.3 Future Work

1. **Fine-tuning on local data.** Fine-tune Chronos-2 on ONS training data and measure the improvement over zero-shot.
2. **Exogenous variables.** Incorporate temperature, humidity, and calendar features using models that support covariates (Chronos-2 supports this via XReg).
3. **Price forecasting.** Extend to CCEE PLD (energy price) forecasting, which is more volatile and may challenge zero-shot transfer.
4. **Longer horizons.** Evaluate week-ahead (168h) forecasting for medium-term planning.
5. **Probabilistic evaluation.** Assess forecast uncertainty using CRPS and calibration metrics, leveraging the quantile outputs these models provide.

---

## 6. Conclusion

We present the first evaluation of time series foundation models on Brazilian electricity load forecasting. Using publicly available data from ONS, we demonstrate that Chronos-2 achieves 1.86% MAPE on day-ahead forecasting for Brazil's largest subsystem — matching the accuracy of proprietary ISO systems in the US and outperforming trained deep learning models on comparable European grids — without any training on Brazilian data. This result holds across all four Brazilian subsystems (1.67-3.17% MAPE, R² > 0.90). We further show that foundation models dominate at operational horizons (24h-168h) but lose to naive seasonal baselines beyond approximately two weeks, identifying a clear practical boundary for zero-shot deployment. Our results provide evidence that the cross-domain transfer capabilities of foundation models extend to emerging market power systems with distinct characteristics (hydro dependency, southern hemisphere seasonality), suggesting a practical path toward accurate STLF in regions where bespoke model development may be resource-constrained.

---

## References

[TODO: Format properly]

- Ansari, A. F., et al. (2024). Chronos: Learning the Language of Time Series. arXiv:2403.07815.
- Das, A., et al. (2024). A decoder-only foundation model for time-series forecasting. ICML 2024.
- Woo, G., et al. (2024). Moirai: A Time Series Foundation Model for Universal Forecasting. ICML 2024.
- NX-AI (2025). TiRex: xLSTM-based Time Series Foundation Model. NeurIPS 2025.
- [arxiv 2602.10848] Foundation models on ERCOT load forecasting.
- [arxiv 2602.05390] Electricity demand forecasting with exogenous data in TSFMs.
- [arxiv 2410.09487] Benchmarking TSFMs for household electricity load forecasting.
- ONS (2021). Portal de Dados Abertos. https://dados.ons.org.br/

---

## Appendix A: Reproducibility

All code and instructions are available at: https://github.com/nelsonbarlow/brazil-energy-forecast

```bash
git clone https://github.com/nelsonbarlow/brazil-energy-forecast.git
cd brazil-energy-forecast
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_ons.py --subsystem SE
python scripts/benchmark.py
```

Hardware: Mac Mini M4, 24GB RAM. No GPU required.
