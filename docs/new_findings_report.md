# Research Progress Report: Negative Binomial Probabilistic Forecasting and Spatial Neighbor Modeling

---

## 1. Overview and Core Breakthroughs

This report documents the architectural advancements and empirical findings developed to address the key challenges identified in the initial AEGIS-Dengue baseline evaluations:

1. **Overcoming the One-Week Persistence Barrier:**Prior evaluations under log-transformed Mean Squared Error (MSE) established that neural models were unable to reliably beat naive persistence ($\hat{y}_{t+1} = y_t$) at a one-week horizon ($h=1$, baseline 16.42 MAE). The introduction of a **count-based Negative Binomial (NB2) likelihood** with origin offset anchoring achieves **15.69 Headline MAE (Ensemble)**, outperforming persistence across all seven headline test folds ($p = 0.0002$).
2. **Resolving the Spatial Graph Dilution Dilemma:**Standard geographic graph convolutions (queen-contiguity and Gaussian distance) previously degraded forecasting performance due to spatial over-smoothing. By replacing graph averaging with explicit **Spatial Neighbor Spillover and Velocity features ($v3$ tensor)**, the model gains border-level transmission awareness without diluting local urban signals, lowering four-week ($h=4$) Peak MAE from 48.71 to 39.92 (-18.0%).
3. **Multi-Horizon Outbreak Prediction ($h=1$ to $h=4$):**Integrating the $v3$ spatial features into a shared-trunk multi-horizon Negative Binomial model establishes superior predictive skill over persistence simultaneously across all four evaluated lead times (1, 2, 3, and 4 weeks ahead).
4. **Calibrated Probabilistic Uncertainty:**
   The model transitions from single point forecasts to full parametric distributions, generating calibrated 80% prediction intervals (10th to 90th percentiles) and exceedance probabilities for operational public health alarms.

---

## 2. Methodology and Architectural Advances

### 2.1 Negative Binomial (NB2) Likelihood Formulation

Epidemiological count series are discrete, non-negative, zero-inflated, and right-skewed with variance substantially exceeding the mean ($\text{Var}(Y) \gg \mathbb{E}[Y]$). Standard MSE in $\log(1 + y)$ space introduces severe nonlinear distortion: a 2-case error in a low-incidence rural district is penalized roughly 14 times more heavily than a 300-case error during an urban epidemic surge.

The Negative Binomial parameterization directly models the observed count data distribution:

$$
\mathbb{E}[Y] = \mu
$$

$$
\text{Var}(Y) = \mu + \alpha \mu^2
$$

where $\mu > 0$ represents the conditional mean count and $\alpha > 0$ denotes the overdispersion parameter. As $\alpha \to 0$, the distribution recovers the standard Poisson process.

#### Epidemiological Origin Offset Anchoring

To preserve the strong baseline signal inherent to the surveillance origin without distorting the parameter scale, the output head parameterizes $\mu$ via an origin offset:

$$
\log(\mu) = \log(1 + y_{\text{origin}}) + \Delta_\mu
$$

$$
\mu = (1 + y_{\text{origin}}) \cdot \exp(\Delta_\mu)
$$

where $y_{\text{origin}}$ is the case count observed at the forecast origin, and $\Delta_\mu$ is constrained to $[-10, 10]$ to prevent numerical overflow. The linear projection weights are initialized to zero, ensuring training begins with the persistence baseline as its prior.

#### Numerical Implementation

The negative log-likelihood (NLL) is computed stably using log-gamma and `log1p` formulations, ensuring full gradient stability on zero observations:

$$
\log P(Y = y \mid \mu, \alpha) = \log \Gamma(y + r) - \log \Gamma(r) - \log \Gamma(y + 1) - (r + y) \log(1 + \alpha \mu) + y \log(\alpha \mu)
$$

where $r = 1 / \alpha$. The masked objective is normalized strictly over observed cells.

---

### 2.2 Spatial Neighbor Spillover and Velocity Features ($v3$ Tensor)

Residual diagnostics established that prediction errors across physically adjacent districts exhibit strong spatial correlation ($r = 0.53\text{--}0.63$). Dense graph convolutions failed because they performed unweighted spatial averaging that blunted sharp local peaks in high-burden urban centers such as Colombo.

The $v3$ tensor extracts the spatial signal as dedicated input channels (28 features total per district-week):

1. **Neighbor Case Mean (`neighbor_cases_log1p_mean`):**The mean $\log(1 + \text{cases})$ across queen-contiguous border neighbors at time $t$, capturing broad regional background transmission.
2. **Neighbor Case Maximum (`neighbor_cases_log1p_max`):**The maximum $\log(1 + \text{cases})$ among contiguous neighbors, providing immediate early warning when an adjacent district enters an exponential surge.
3. **Neighbor Case Velocity (`neighbor_case_velocity`):**
   The week-over-week difference in mean neighbor counts ($\text{Mean}_{t} - \text{Mean}_{t-1}$), distinguishing accelerating outbreaks from subsiding waves.

All neighbor features are strictly causal: they read only contemporary ($t$) and historical ($t-1$) observations, avoiding any forward leakage into target periods.

---

### 2.3 Shared Multi-Horizon Architecture

The model uses a shared Gated Recurrent Unit (GRU) trunk over 12-week lookback windows, feeding into four distinct Negative Binomial heads predicting horizons $h \in \{1, 2, 3, 4\}$. This joint multi-task formulation allows near-horizon supervisory signals ($h=1$) to stabilize representations for longer-horizon predictions ($h=4$), while operating with one-quarter of the trunk parameter count required by separate per-horizon models.

---

## 3. Empirical Results and Performance Progression

### 3.1 Multi-Horizon Benchmark Across All Headline Folds

All models were evaluated using the standardized 9 walk-forward folds. The headline metrics represent the mean across the seven non-COVID test folds (2017, 2018, 2019, 2022, 2023, 2024, 2025):

| Model Stage            | Model Configuration                                    | $h=1$ (1 wk) | $h=2$ (2 wks) | $h=3$ (3 wks) | $h=4$ (4 wks) | $h=4$ Peak MAE |
| :--------------------- | :----------------------------------------------------- | :-------------: | :-------------: | :-------------: | :-------------: | :--------------: |
| **Reference**    | **Naive Persistence Baseline**                   | **16.42** | **20.23** | **24.81** | **28.78** | **48.71** |
| Initial Baseline       | Geographic Contiguity Shared (v1, MSE)                 |      18.19      |      22.31      |      25.83      |      28.65      |      49.52      |
| Initial Baseline       | Identity Control Shared (v1, MSE)                      |      16.71      |      20.04      |      23.15      |      25.74      |      45.20      |
| Intermediate           | Layer Normalization (v1, MSE)                          |      16.64      |      19.98      |      23.01      |      25.53      |      44.80      |
| Intermediate           | Outbreak History Features (v2, MSE)                    |      16.39      |      19.57      |      22.64      |      25.15      |      43.10      |
| Probabilistic          | Negative Binomial Single-Horizon (v1, Ens)             |      15.69      |       —       |       —       |       —       |      25.32      |
| **Current Best** | **Multi-Horizon NegBin on $v3$ (Single-Seed)** | **16.09** | **19.44** | **22.91** | **25.85** | **40.35** |
| **Current Best** | **Multi-Horizon NegBin on $v3$ (Ensemble)**    | **15.88** | **19.18** | **22.59** | **25.44** | **39.92** |

Key observations:

* At **$h=1$**, the model scores **15.88 MAE (Ensemble)** vs 16.42 for persistence, establishing a consistent win across short horizons.
* At **$h=4$**, the model scores **25.44 MAE (Ensemble)** vs 28.78 for persistence, representing an improvement of +11.6% overall and +18.0% during epidemic peak weeks.

---

### 3.2 Performance on Peak Transmission (Peak MAE)

Peak MAE isolates prediction errors specifically during the highest-incidence weeks within each district's historical record:

|              Horizon              | Persistence Peak MAE | Multi-Horizon NegBin ($v3$) Peak MAE | Absolute Reduction | Relative Gain |
| :-------------------------------: | :------------------: | :------------------------------------: | :----------------: | :-----------: |
| **$h=1$** (1 week ahead) |        26.61        |            **25.19**            |  -1.42 cases/week  |     +5.3%     |
| **$h=2$** (2 weeks ahead) |        33.04        |            **30.15**            |  -2.89 cases/week  |     +8.7%     |
| **$h=3$** (3 weeks ahead) |        41.93        |            **36.08**            |  -5.85 cases/week  |    +13.9%    |
| **$h=4$** (4 weeks ahead) |        48.71        |            **39.92**            |  -8.79 cases/week  |    +18.0%    |

The predictive margin expands monotonically with the lead time. At a four-week lead time, where persistence degrades significantly, the model reduces outbreak peak errors by nearly 9 cases per district-week.

---

### 3.3 The 2017 National Epidemic Benchmark (Fold 1)

Fold 1 evaluated models on the unprecedented 2017 national epidemic (>175,000 cases nationally):

| Model Architecture                                  | Objective Function       | 2017 Test MAE ($h=1$) |         Relative to Persistence         |
| :-------------------------------------------------- | :----------------------- | :---------------------: | :-------------------------------------: |
| Baseline GCN+GRU (v1)                               | Masked Log-MSE           |          54.81          |    -51.9% (Severe under-prediction)    |
| Identity GRU-Only (v1)                              | Masked Log-MSE           |          42.97          |                 -19.1%                 |
| Level-Weighted GCN+GRU (v1)                         | Weighted Log-MSE         |          38.06          |                  -5.5%                  |
| **Naive Persistence Reference**               | —                       |     **36.08**     |                Baseline                |
| Intermediate Outbreak History (v2)                  | Masked Log-MSE           |      39.00–45.60      |                  -8.1%                  |
| Negative Binomial Single-Horizon (v1)               | NB2 Likelihood           |     **35.47**     |                  +1.7%                  |
| **Multi-Horizon NegBin on $v3$ (Ensemble)** | **NB2 Likelihood** |     **34.78**     | **+3.6% (Lowest Error Recorded)** |

Under the Negative Binomial likelihood, the model avoids under-predicting the exponential surge, achieving the lowest test error recorded for the 2017 epidemic.

---

## 4. Statistical Significance Testing

Paired hypothesis tests were conducted across the seven headline walk-forward test folds (degrees of freedom = 6):

```text
Paired Comparison: Negative Binomial Ensemble vs. Naive Persistence (h=1)
  Model Mean:        15.69 MAE
  Persistence Mean:  16.42 MAE
  Mean Difference:   -0.74 MAE (+4.5% overall skill)
  Headline Folds Won: 7 of 7 folds
  Paired t-test:     t = -8.253, p = 0.0002 (Statistically significant at p < 0.001)
  Wilcoxon Test:     W = 0.0, p = 0.0156 (Non-parametric significance confirmed)
  Effect Size:       Cohen's d = -3.12 (Very large effect size)
```

The uniform superiority across all seven headline folds confirms that the gain is not an artifact of outlier test periods.

---

## 5. Uncertainty Quantification and Probabilistic Outputs

Unlike deterministic regression architectures, the fitted Negative Binomial parameters $(\mu, \alpha)$ enable rigorous probabilistic forecasting:

1. **Calibrated Prediction Intervals:**Parametric quantiles derived via the negative binomial percent-point function:

   $$
   \hat{y}_{q} = F^{-1}_{\text{NB}}(q \mid \mu, \alpha)
   $$

   The generated publication figure (`figures/fig_probabilistic_forecast.png`) illustrates observed cases alongside the predicted mean and shaded 80% intervals ($q=0.10$ to $q=0.90$) for Colombo and Gampaha throughout the 2017 epidemic.
2. **Epidemic Alarm Exceedance Probabilities:**
   The probability that a district will exceed an operational outbreak threshold $T$ during week $t+h$:

   $$
   P(Y_{t+h} \ge T) = 1 - F_{\text{NB}}(T - 1 \mid \mu_{t+h}, \alpha_{t+h})
   $$

   This capability allows public health authorities to trigger actionable, risk-calibrated early warnings rather than relying on uncalibrated point estimates.

---

## 6. Reproducibility and Code Execution

All models and experiments are implemented as standalone, modular components:

* **Tensors and Preprocessing:**
  ```powershell
  python scripts/features/12.build_model_tensors.py
  python scripts/features/14.build_folds.py
  ```
* **Single-Horizon NegBin Training ($h=1$):**
  ```powershell
  python scripts/training/31.train_negbin.py --variant v3 --seeds 3
  ```
* **Multi-Horizon NegBin Training ($h=1..4$):**
  ```powershell
  python scripts/training/32.train_multi_horizon_negbin.py --variant v3 --seeds 3
  ```
* **Significance Testing Suite:**
  ```powershell
  python scripts/evaluation/34.evaluate_statistical_significance.py
  ```
* **Probabilistic Visualizations:**
  ```powershell
  python scripts/evaluation/33.plot_probabilistic_forecasts.py
  ```

Outputs and results tables are stored under `results/models/` and `results/tables/`.
