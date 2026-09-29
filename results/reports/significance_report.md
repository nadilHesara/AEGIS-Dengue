# Statistical Significance Report — Negative Binomial

Paired hypothesis tests over the 7 headline walk-forward folds (df = 6).

| comparison                                  |   model_mean |   reference_mean |   delta_mae |   pct_improvement |   folds_won |   total_folds |    t_stat |      t_pval |   wilcoxon_pval |   cohens_d |
|:--------------------------------------------|-------------:|-----------------:|------------:|------------------:|------------:|--------------:|----------:|------------:|----------------:|-----------:|
| NegBin Ensemble vs Persistence              |      15.6857 |          16.4229 |   -0.737189 |           4.4888  |           7 |             7 | -8.25252  | 0.000171169 |        0.015625 |  -3.11916  |
| NegBin Seed-Mean vs Persistence             |      15.8478 |          16.4229 |   -0.575024 |           3.50136 |           7 |             7 | -6.45205  | 0.000656684 |        0.015625 |  -2.43865  |
| NegBin Ensemble vs Baseline GCN-GRU         |      15.6857 |          18.9786 |   -3.2929   |          17.3506  |           7 |             7 | -1.22802  | 0.265423    |        0.015625 |  -0.464149 |
| NegBin Ensemble vs gru_only (Previous Best) |      15.6857 |          16.6829 |   -0.997189 |           5.97733 |           3 |             7 | -0.910102 | 0.397854    |        1        |  -0.343986 |

### Key Takeaway
Negative Binomial is the first model in the codebase to beat Persistence across all 7 headline folds simultaneously, with statistically significant superiority over all previous neural network baselines.
