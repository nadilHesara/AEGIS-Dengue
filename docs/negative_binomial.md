# Negative Binomial Probabilistic Head — Breaking the Horizon-1 Persistence Barrier

A standalone, principled count-likelihood implementation (Component C) that replaces
log-space MSE with a Negative Binomial (NB2) maximum-likelihood objective, evaluated
on the frozen 9-fold walk-forward benchmark.

Reproduce with:
```powershell
python scripts/training/31.train_negbin.py --seeds 3
```
Associated modules and tests:
- Model & Loss: [`src/models/negative_binomial.py`](../src/models/negative_binomial.py)
- Training Script: [`scripts/training/31.train_negbin.py`](../scripts/training/31.train_negbin.py)
- Unit Tests: [`tests/test_negative_binomial.py`](../tests/test_negative_binomial.py) (7 tests, all passing)
- Metrics Output: `results/models/negbin_metrics_identity_v1_h1.csv`

---

## 1. Background & The Diagnosed Objective Mismatch

Throughout the repository's history, every model configuration tested at horizon 1
failed to beat the naive persistence baseline ($\hat{y}_{t+1} = y_t$, Headline MAE 16.42):
- Baseline GCN+GRU: **18.98 MAE**
- Identity control (`gru_only`): **16.68 MAE**
- Level-weighted MSE loss: **16.59 MAE** (ensemble tied persistence at 16.36 within seed noise)

### The Underlying Mathematical Cause
The baseline trained on **masked MSE over the log1p residual**:
$$d = \log(1 + y_{t+1}) - \log(1 + y_t)$$
$$\mathcal{L}_{\text{MSE}} = (d - \hat{d})^2$$
while evaluation is scored on **raw case-count MAE**:
$$\text{MAE} = \frac{1}{N} \sum |y - \hat{y}|$$

This created two critical distortions:
1. **Gradient Misallocation:** A constant 0.20 log-space error is 1.3 cases when $y=5$, but 332 cases when $y=1,500$. Squared log-error spends equal gradient on both, whereas MAE penalizes the epidemic surge 250× more heavily.
2. **Under-prediction of Epidemics:** The model systematically under-predicted explosive surges (predicting ~60% of peak volume in the 2017 epidemic), inflating Fold 1 MAE to 42–55 against persistence's 36.08.

---

## 2. Mathematical Formulation

Dengue surveillance data is non-negative, discrete, zero-inflated, and exhibits severe
overdispersion ($\text{Var}(Y) \gg \mathbb{E}[Y]$). The Negative Binomial (NB2)
distribution parameterizes this directly without ad-hoc log transforms:

$$\mathbb{E}[Y] = \mu$$
$$\text{Var}(Y) = \mu + \alpha \mu^2$$

where:
- $\mu > 0$ is the conditional mean case count.
- $\alpha > 0$ is the overdispersion parameter ($r = 1/\alpha$ is the shape parameter).
- When $\alpha \to 0$, the distribution converges to Poisson ($\text{Var}(Y) = \mu$).

### Numerically Stable Negative Log-Likelihood (NLL)
To ensure numerical stability across zero counts and massive epidemic spikes,
the log-likelihood is computed using `torch.log1p` and `torch.lgamma`:

$$\log P(Y = y \mid \mu, \alpha) = \log \Gamma(y + r) - \log \Gamma(r) - \log \Gamma(y + 1) - (r + y) \log(1 + \alpha \mu) + y \log(\alpha \mu)$$

For unobserved cells, the loss is masked out and normalized by the number of observed cells:

$$\mathcal{L}_{\text{NegBin}} = \frac{\sum_{i, t} \text{mask}_{i, t} \cdot \left(-\log P(Y_{i, t} = y_{i, t} \mid \mu_{i, t}, \alpha_{i, t})\right)}{\sum_{i, t} \text{mask}_{i, t}}$$

---

## 3. Architecture & Epidemiological Offset Anchoring

The model architecture retains the baseline's spatio-temporal GRU sequence trunk,
replacing only the output layer with `NegBinHead` (`src/models/negative_binomial.py`):

1. **Epidemiological Offset Anchoring for Mean ($\mu$):**
   $$\log(\mu) = \log(1 + y_{\text{origin}}) + \Delta_\mu$$
   $$\mu = (1 + y_{\text{origin}}) \cdot \exp(\Delta_\mu)$$
   $\Delta_\mu$ is clamped to $[-10, 10]$ to prevent float exponent overflow.
   The projection layer weights and bias are initialized to zero, ensuring the model
   **starts training exactly at the persistence baseline** ($\Delta_\mu = 0 \implies \mu \approx y_{\text{origin}}$).

2. **Dynamically Learned Overdispersion ($\alpha$):**
   $$\alpha = \text{softplus}(W_\alpha h + b_\alpha) + 10^{-4}$$
   This guarantees $\alpha > 0$ strictly, allowing the model to adaptively expand its
   variance window during volatile epidemic weeks and contract it during quiet endemic periods.

---

## 4. Full 9-Fold Benchmark Results ($h=1$, 3 Seeds)

Evaluated under the exact walk-forward protocol (9 folds, seeds 0, 1, 2, identity backbone, $v1$ features):

| Model / Configuration | Headline MAE | vs Persistence | Peak MAE | 2017 Epidemic MAE | Headline Folds Won |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline `gcn_gru` v1 (MSE)** | 18.98 | $-15.6\%$ | 30.36 | 54.81 | 0 / 7 |
| **Previous Best `gru_only` v1 (MSE)** | 16.68 | $-1.6\%$ | 28.06 | 42.97 | 5 / 7 |
| **Level-Weighted MSE (`scripts/20`)** | 16.59 | $-1.0\%$ | 27.47 | 38.06 | 5 / 7 |
| **Persistence Baseline** | **16.42** | — | **26.61** | **36.08** | — |
| **NegBin (Single-Seed Mean)** | **15.85** | **$+3.5\%$** | **25.54** | **35.80** | **7 / 7** |
| **NegBin (Seed-Mean Ensemble)** | **15.69** | **$+4.4\%$** | **25.32** | **35.47** | **7 / 7** |

---

## 5. Fold-by-Fold Performance

Comparing the Negative Binomial ensemble against the persistence baseline:

| Fold | Test Year | Characteristics | Persistence MAE | NegBin Ensemble MAE | $\Delta$ (Cases/Week) | Result |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **Fold 1** | **2017** | **National Epidemic (>175k cases)** | 36.08 | **35.47** | **$-0.61$** | **WON** |
| **Fold 2** | 2018 | Post-epidemic lull | 10.45 | **9.95** | **$-0.50$** | **WON** |
| **Fold 3** | 2019 | Moderate endemic surge | 17.30 | **16.39** | **$-0.91$** | **WON** |
| **Fold 4** | 2020 | COVID lockdown (separate) | 7.41 | 7.74 | $+0.33$ | — |
| **Fold 5** | 2021 | COVID lockdown (separate) | 7.23 | 7.09 | $-0.14$ | — |
| **Fold 6** | 2022 | Post-lockdown recovery | 12.57 | **12.08** | **$-0.49$** | **WON** |
| **Fold 7** | 2023 | Regional transmission surge | 18.83 | **17.75** | **$-1.08$** | **WON** |
| **Fold 8** | 2024 | Recent normal year | 10.78 | **9.84** | **$-0.94$** | **WON** |
| **Fold 9** | 2025 | Recent normal year | 8.95 | **8.33** | **$-0.62$** | **WON** |

### Key Findings:
1. **Unanimous Fold Superiority:** The model beats persistence on **all 7 out of 7 headline folds**. The headline gain is uniformly distributed across normal and surge years, rather than being an artefact of a single fold.
2. **2017 Epidemic Victory:** For the first time in the repository, a trained model beats persistence during the 2017 national epidemic (**35.47 vs 36.08**).
3. **Peak MAE Victory at $h=1$:** Achieves **25.32 Peak MAE** (ensemble), beating persistence's **26.61**.
4. **Seed Variance Stability:** Across all folds, the seed standard deviation is small ($\sigma \approx 0.3 - 0.5$), confirming smooth and stable convergence on CPU.

---

## 6. Probabilistic Forecasting Capabilities

Unlike point-prediction regression heads, the fitted Negative Binomial parameters $(\mu, \alpha)$ enable:
1. **Prediction Intervals:** Exact quantile computation via `scipy.stats.nbinom.ppf`:
   ```python
   from src.models.negative_binomial import compute_prediction_intervals
   intervals = compute_prediction_intervals(mu, alpha, quantiles=(0.1, 0.5, 0.9))
   ```
2. **Outbreak Early-Warning Probabilities:** Exceedance probability above an operational epidemic threshold $T$:
   $$P(Y \ge T) = 1 - F_{\text{NB}}(T - 1 \mid \mu, \alpha)$$
   ```python
   from src.models.negative_binomial import compute_outbreak_probability
   alarm_prob = compute_outbreak_probability(mu, alpha, threshold=100.0)
   ```
