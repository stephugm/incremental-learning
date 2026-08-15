# Thesis Plan — MSc in Artificial Intelligence

## Title
**"Is It Skill or Smoothing? A Cost-Aware Re-Evaluation of Incremental Deep Learning for Real-Time Stock Index Forecasting"**

---

## 1. Background & Motivation

### Base Paper
Kalra et al. (2024). *An efficient hybrid approach for forecasting real-time stock market indices.* Journal of King Saud University – Computer and Information Sciences, 36, 102180.

**What it does:**
- Proposes H.BLSTM — a bidirectional LSTM with incremental updates during each 15-minute trading bar plus a full batch retrain on all accumulated data at the end of every session.
- Uses univariate (close price only) and multivariate (close + EMA5) inputs across nine major global indices (DJI, SPX, NDX, DXY, NI225, SSE, NIFTY, FTSE, DAXEUR).
- Reports MAPE ≈ 0.001 (average across nine indices) and an inference latency of ~2 s.

### Supporting Paper
Qian (2025). *An enhanced Transformer framework with incremental learning for online stock price prediction.* PLOS ONE, 20(1): e0316955.

**What it adds:**
- IL-ETransformer: multi-head self-attention + continual normalisation to handle non-stationarity + Time-Series Elastic Weight Consolidation (TS-EWC) to prevent catastrophic forgetting.
- Notes explicitly that incremental-learning research focuses on classification; financial regression is underexplored.
- Evaluates on five daily-frequency public datasets; no real-time/HFT deployment, no technical-indicator engineering, no latency measurement.

---

## 2. Why the Results Look "Too Good" — The Actual Problem

### 2.1 The Naive Baseline Was Never Tested
At a 15-minute forecasting horizon, typical index return volatility is approximately 0.1–0.25% per bar. The H.BLSTM's reported MAPE of 0.001 (~0.1–0.2% actual error) sits squarely inside the noise floor of the series. A naive persistence forecast — **"next price = current price"** — achieves essentially the same error because the signal-to-noise ratio at 15 minutes is extremely low.

The paper's baselines (Linear Regression, KNN, Decision Tree) do not test this. Notably, KNN — which memorises the last 50 samples and approximates persistence — already achieves MAPE 0.002, only a factor of 2 away from H.BLSTM. This is the classic fingerprint of a model that has learned little more than "copy the last close."

**If the base paper had included a naive persistence baseline, the headline claim of "significantly outperforming traditional models" might not hold.**

### 2.2 The Multivariate Gain May Be Smoothing, Not Information
The paper's *entire* multivariate claim rests on **one indicator: EMA5**, a 5-period exponential moving average of the close price. EMA5 is a smoothed transform of the target itself — it carries almost no information that is not already present in the raw close window. The paper itself admits that for ML models (LR/KNN/DT) there was "insignificant change in average MAPE between univariate and multivariate," which is exactly what you would expect if EMA5 adds no new information. For H.BLSTM, the gain may simply be the model receiving a pre-smoothed target that reduces high-frequency noise — not genuine multivariate signal.

**Experiment to test this:** replace EMA5 with an information-free low-pass filter (e.g., a causal moving average). If the "gain" persists, it was smoothing artefact, not information. This is the sharpest single experiment in the thesis.

### 2.3 The Evaluation Protocol Has Known Flaws
- No error bars, no multiple seeds, no confidence intervals.
- MAPE on price levels can be gamed: a model that always predicts yesterday's level scores near-zero MAPE.
- No directional accuracy, no statistical significance testing (Diebold–Mariano), no economic validation (backtest, Sharpe, hit rate).
- No out-of-regime testing (COVID, 2022 drawdown, flash crashes).
- Possible data-leakage risk: scaling and EMA computed over the full series before splitting — the paper does not confirm train-only scaler fitting.

### 2.4 The Update Cost Is Hidden
H.BLSTM is described as "partially incremental" — it requires a full batch retrain on **all accumulated data** at the end of every trading session. This cost grows unboundedly with time and instrument count. Yet the paper reports only inference latency (2 s), never the update/retrain cost. At 50 instruments × 200 days of 15-min bars each, this is a serious scalability problem that the paper does not acknowledge.

---

## 3. Research Questions

| # | Research Question |
|---|---|
| **RQ1** | Does H.BLSTM outperform a naive persistence baseline on 15-minute index forecasting when evaluated with MASE, directional accuracy, and a Diebold–Mariano significance test? |
| **RQ2** | Is the reported EMA5 multivariate gain genuine predictive information or an input-smoothing artefact? Do eight diverse technical indicators (trend, momentum, volatility, volume) change the result? |
| **RQ3** | Can a drift-triggered retraining policy match the fixed end-of-session full retrain at materially lower cumulative update cost, measured by wall-clock time and peak memory? |

---

## 4. Hypotheses

- **H1:** H.BLSTM's MASE will be ≥ 0.90 (near 1.0), indicating no meaningful skill over naive persistence. (Falsifiable: if MASE < 0.85, the base paper's claim is validated.)
- **H2:** Replacing EMA5 with an information-free low-pass filter will produce statistically equivalent or higher accuracy, revealing that EMA5's benefit is smoothing, not information. An 8-indicator diverse set will achieve MASE significantly below EMA5-only.
- **H3:** Drift-triggered retraining will reduce total retraining events by ≥ 40% relative to the fixed-schedule policy while maintaining statistically equivalent accuracy (DM test fails to reject H₀ of equal forecast accuracy).

---

## 5. Methodology

### 5.1 Data
- **Indices:** SPX (USA), NIFTY (India), FTSE (UK) — one per region, selected to represent different volatility regimes and trading-session overlaps.
- **Frequency:** 15-minute bars (OHLCV).
- **Source:** tvDatafeed (TradingView) as in the base paper; fallback: yfinance (limited to 60 days per pull → rolling acquisition strategy).
- **Volume:** ≥ 6,000 bars per index (matching base paper), from at least two market regimes.
- **Preprocessing:** train/validation/test split in strict chronological walk-forward order. **Scaler fit on train only, applied to val/test.** No shuffling.
- **Regime annotation:** label each week as high-volatility or low-volatility using a 5-period rolling ATR percentile. Report metrics separately per regime.

### 5.2 Model Implementation

**H.BLSTM (reproduced exactly as described):**
- 2-layer bidirectional LSTM.
- Incremental within-session: parameters updated on each new 15-min bar using Adam with a small learning rate.
- End-of-session batch retrain: full re-fit on all accumulated historical data.
- Hyperparameters: use base paper's reported values (Adam, MSE loss, ReLU, max 50 epochs, grid-searched config from Fig. 4). Do **not** run a new grid search.

**Baselines (all run in walk-forward mode):**
- Naive persistence: $\hat{y}_{t+1} = y_t$
- Seasonal naive: $\hat{y}_{t+1} = y_{t-N}$ where $N$ = number of bars per day (same time yesterday)
- Drift: $\hat{y}_{t+1} = y_t + (y_t - y_1)/t$
- Online Linear Regression (`river` library, replicates base paper)
- Online KNN (`river`, last-50-instance window)
- Online Decision Tree (`river` HATR, replicates base paper)

### 5.3 Evaluation Protocol (RQ1)

**Metrics:**

| Metric | Formula | Why |
|---|---|---|
| **MASE** | $\frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{naive}}}$ | < 1 = skill; ≈ 1 = no skill. Cannot be gamed by price scale. |
| **Directional accuracy** | $\frac{1}{n}\sum \mathbf{1}[\text{sign}(\hat{r}_t) = \text{sign}(r_t)]$ | Whether the model can predict up/down moves. Random = 50%. |
| **MAPE** | as in base paper | For direct comparability. |
| **RMSE, MAE** | standard | For completeness. |
| **Diebold–Mariano test** | DM statistic with Harvey et al. small-sample correction | Statistical significance of error differences. |

**Protocol:**
- Minimum 3 independent random seeds for stochastic models.
- Report mean ± std across seeds.
- Evaluate on the **full test period** and separately on **high-volatility** vs **low-volatility** regime sub-windows.
- DM test: H.BLSTM vs naive persistence. p < 0.05 required to claim skill.

### 5.4 Technical-Indicator Study (RQ2)

**Indicator library (8 indicators, all implemented in incremental O(1) form):**

| Family | Indicator | Update complexity |
|---|---|---|
| Trend | EMA20 | O(1) |
| Trend | MACD signal line (EMA12 – EMA26) | O(1) |
| Momentum | RSI(14) via Wilder smoothing | O(1) |
| Momentum | Rate of Change (ROC-5) | O(w), w=5, acceptable |
| Volatility | ATR(14) via Wilder smoothing | O(1) |
| Volatility | Bollinger Band %B (20-period) | O(w), w=20 |
| Volume | OBV (on-balance volume) | O(1) |
| Volume | Volume ratio (bar vol / 20-bar avg vol) | O(w), w=20 |

**Ablation conditions:**

| Condition | Feature set | Purpose |
|---|---|---|
| A0 | Close only | Univariate baseline |
| A1 | Close + EMA5 | Exact base-paper multivariate setup |
| A2 | Close + low-pass filter (causal MA-5, no EMA weights) | **Smoothing-control experiment** — same smoothing as EMA5 but zero information content |
| A3 | Close + 8 indicators above | Diverse multivariate |
| A4 | Close + top-4 by permutation importance | Cost-efficient subset |

**Key test:** if A1 and A2 produce statistically equivalent MASE (DM test fails to reject), EMA5's benefit is smoothing artefact. If A3 significantly beats A2 and A1 (DM p < 0.05), genuine multivariate gain exists.

**Feature importance:** run permutation importance over three temporal windows (early / mid / late test period) to produce a simple feature-stability table showing which indicators retain importance across regimes.

### 5.5 Drift-Triggered Retraining (RQ3)

**Drift detector:** ADWIN (Adaptive Windowing, `river` library) applied to the rolling absolute forecast error. When ADWIN signals drift, trigger a retrain. Also evaluate Page-Hinkley as a lighter alternative.

**Update mechanisms compared:**

| Mechanism | Description |
|---|---|
| **Fixed (base paper)** | Full retrain on all data at end of every trading session |
| **Drift-triggered** | Full retrain only when ADWIN/Page-Hinkley fires on residuals |
| **TS-EWC (borrowed from paper 2)** | EWC regularisation on weight updates; no full retrain needed |

**Cost metrics:**
- Total number of retrains triggered (× all sessions)
- Cumulative wall-clock retrain time (seconds, on fixed hardware)
- Peak memory (MB) of the retrain buffer
- Accuracy at end of each session (MASE, directional accuracy)

**Pareto plot:** x-axis = cumulative update cost (wall-clock seconds); y-axis = test MASE. Each mechanism is a point. Pareto-optimal = lowest MASE at lowest cost.

---

## 6. Expected Contributions

1. **First rigorous evaluation of H.BLSTM against naive persistence** with MASE and Diebold–Mariano testing — confirming or refuting whether the reported gains reflect genuine forecasting skill.
2. **Experimental proof or refutation of the EMA5-as-smoothing hypothesis**, via a controlled comparison with an information-free low-pass filter. This directly validates or invalidates a claim in a published paper.
3. **An 8-indicator incrementally-computable multivariate feature set** with per-indicator update-cost profiling, and evidence on which families survive regime change.
4. **First published update-cost curve for hybrid incremental financial forecasting** (cost vs. data volume), and a drift-triggered policy that reduces retraining frequency while maintaining statistical parity with the fixed-schedule baseline.
5. **An accuracy–cost Pareto plot** for three update mechanisms, providing concrete deployment recommendations.

### Why this thesis cannot "fail"
- If H.BLSTM beats naive persistence (MASE < 1, DM significant) → **contributions 3–5 stand alone**.
- If H.BLSTM does not beat naive persistence → **contribution 1 is itself a publishable finding**; contributions 2–5 still stand.
- RQ3 succeeds on a "no-worse-accuracy at lower cost" criterion, which is easier to achieve than "better accuracy." Even a null result (drift-triggered = fixed, same cost) is informative.

---

## 7. Scope Decisions and Reasoning

| Decision | Chosen | Reason |
|---|---|---|
| Number of indices | 3 (SPX, NIFTY, FTSE) | One per region covers diversity; 9 indices × same experiments = 3× effort with marginal new insight. State 9-index replication as future work. |
| Backbone | H.BLSTM only (no IL-ETransformer implementation) | Implementing two complex models in 4 months is infeasible. Borrow only TS-EWC algorithm from paper 2. |
| Hyperparameter search | Use base paper's values | New grid search alone can consume 40 h. H.BLSTM's config is published; use it and document this. |
| Evaluation scope | MASE + directional accuracy + DM test; no full trading backtest | A full live-trading backtest (transaction costs, order-book simulation) adds 40+ h. The simpler DM test and directional accuracy are sufficient to validate skill. Leave full backtest as future work. |
| FLOP/energy measurement | Excluded | Instrumentation is finicky; wall-clock + memory are sufficient for the cost story and are reproducible. |
| SHAP drift map | Simplified to permutation importance × 3 windows | Full SHAP on a BiLSTM is expensive; permutation importance achieves 80% of the insight at 10% of the cost. |
| Indicators | 8 (not 25) | 2 per family; each family is represented; ablation is clean; compute is manageable. |
| LoRA / replay | Excluded, cited as future work | Each mechanism adds ~20 h of implementation + experiment time. TS-EWC + drift-triggered is sufficient for RQ3. |

---

## 8. 17-Week Schedule (~16 h/week, total ~270 h)

| Week | Work | Deliverable |
|---|---|---|
| **1** | Literature review (focused: ~30 papers). Finalise proposal. Set up repo, coding environment, experiment tracking (MLflow or W&B lite). | `plan.md` (this document), annotated bibliography |
| **2** | Data pipeline: download 3 indices at 15-min resolution, clean, validate, build walk-forward split framework, confirm no future leakage. | `data/` folder, data quality report |
| **3–4** | H.BLSTM reproduction: implement BiLSTM + within-session incremental update + end-of-session retrain. Validate against paper's reported numbers on SPX. | Reproduced MAPE ≈ 0.001 on SPX |
| **5** | Add all baselines: naive persistence, seasonal naive, drift, online LR, online KNN, online DT. Run all on 3 indices. | Baseline results table |
| **6–7** | Evaluation module: MASE, directional accuracy, DM test, ≥3 seeds, regime annotation. Run RQ1 experiments on all 3 indices. Write Chapter 3 (Evaluation Results). | **RQ1 answered** |
| **8–9** | Implement 8-indicator library in incremental form. Run ablation conditions A0–A4 on all 3 indices. Profile per-indicator compute. | Feature ablation table |
| **10** | Smoothing-control experiment (A1 vs A2). Permutation importance × 3 windows. Write Chapter 4 (Indicator Study). | **RQ2 answered** |
| **11–12** | Implement ADWIN/Page-Hinkley drift detector. Implement drift-triggered retraining policy. Implement TS-EWC. Log cost metrics. | Drift-triggered system running |
| **13** | Run RQ3 experiments on 3 indices. Build Pareto plot. Write Chapter 5 (Cost-Aware Retraining). | **RQ3 answered** |
| **14–15** | Write Chapters 1 (Introduction), 2 (Literature Review), 6 (Conclusion, Limitations, Future Work). | Full draft |
| **16** | Revisions, produce all final figures, proofread. | Final draft |
| **17** | Buffer / supervisor feedback / defence prep. | Submission-ready |

### Critical path rule
RQ1 is answered by **week 7**. If weeks 8–13 slip, you still have a defensible two-RQ thesis (RQ1 + RQ2). Do not start writing only at the end — draft each chapter immediately after its experiments complete.

### Hard scope-freeze: Week 8
No new features, no new experiments, no new models after week 8. If a new idea emerges, add it to the "future work" section.

---

## 9. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| 15-min historical data unavailable for full period | Medium | High | Use rolling 60-day pulls from yfinance + tvDatafeed; or reduce to 3,000 bars instead of 6,000 |
| Cannot reproduce base paper's exact MAPE | Medium | Low | Document discrepancy honestly; it is a finding, not a failure |
| H.BLSTM training is very slow on CPU | Medium | Medium | Reduce model size (1 BiLSTM layer, smaller hidden dim); 3 indices × small model is CPU-feasible |
| All three RQ experiments take longer than scheduled | Medium | High | Drop RQ3 if week 10 is already at risk; two-RQ thesis is fully valid |
| Drift detector fires too rarely (near zero retrains) | Low | Medium | Tune ADWIN sensitivity parameter; report the sensitivity analysis |
| Scope creep from new ideas | High | High | Hard scope-freeze at week 8; write new ideas into future work section |

---

## 10. Tools and Libraries

| Purpose | Tool |
|---|---|
| Data download | `tvDatafeed`, `yfinance` |
| Data processing | `pandas`, `numpy` |
| Incremental ML baselines | `river` (sklearn-river) |
| Deep learning (H.BLSTM) | `PyTorch` or `TensorFlow/Keras` |
| Technical indicators (incremental) | Custom implementation (see WP3) |
| Drift detection | `river` (ADWIN, Page-Hinkley) |
| Experiment tracking | MLflow (local) |
| Statistical tests | `scipy.stats`, `arch` (DM test) |
| Visualisation | `matplotlib`, `seaborn` |
| Version control | `git` |

---

## 11. Thesis Chapter Outline

1. **Introduction** — Motivation, problem statement, research questions, contributions, thesis structure.
2. **Literature Review** — Stock market forecasting (LSTM, BiLSTM, Transformer), incremental/continual learning, technical indicators, evaluation methodology critique.
3. **Methodology** — Data, H.BLSTM architecture, baselines, evaluation protocol (MASE, DM test, directional accuracy), indicator library, update mechanisms.
4. **RQ1 — Evaluation Results** — H.BLSTM vs naive persistence on 3 indices; regime breakdown; interpretation.
5. **RQ2 — Technical Indicator Study** — Ablation A0–A4; EMA5 smoothing-control experiment; feature-importance stability table.
6. **RQ3 — Cost-Aware Retraining** — Update-cost curves; drift-triggered vs fixed vs TS-EWC; Pareto plot; deployment recommendations.
7. **Conclusion** — Findings summary, limitations, future work (9-index replication, full backtest, LoRA updates, IL-ETransformer comparison, conformal prediction intervals).

---

## 12. Future Work (explicitly deferred)

- Expand from 3 to 9 indices as in the base paper.
- Full economic backtest with transaction costs, bid-ask spread, and slippage.
- IL-ETransformer reproduction and head-to-head comparison.
- LoRA/adapter-based parameter-efficient updating.
- Experience replay (ER, DER++) as an alternative to TS-EWC.
- Conformal prediction intervals for uncertainty-aware trading.
- SHAP-based feature attribution and full regime-drift map.
- Scaling study: 3 → 50 → 200 instruments simultaneously.