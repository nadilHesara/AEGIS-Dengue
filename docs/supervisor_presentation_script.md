# AEGIS Dengue — Supervisor Presentation Script

## 1. About the research

### Problem statement

Dengue cases in Sri Lanka change strongly from one reporting period to another, and different districts can show different patterns. Weather variables such as rainfall, temperature, humidity and dewpoint may affect dengue transmission, but their effect may appear after a delay. At the same time, districts that are not physical neighbours can still show similar dengue behaviour.

The main research question is:

> **How can we use recent dengue cases, climate information and relationships between districts to improve short-term dengue forecasting?**

The current system forecasts the next reporting period for all 25 Sri Lankan districts. A major challenge is unusual outbreaks: the 2017 epidemic is much harder than normal years because the model has to forecast a case level not seen in its training period.

### Planned improvements

1. **Learnable climate lag encoder** — learn which previous climate periods are useful for each district and climate variable instead of fixing the same delays manually.
2. **Data-driven district graph** — identify districts with similar dengue case patterns instead of relying only on shared geographical borders.
3. **Improved spatial information** — test whether the data-driven relationships provide more useful information than the current geographical graph.

The overall idea is to learn relationships that are currently fixed by hand.

---

# 2. Current baseline: GCN + GRU

The baseline uses a **Graph Convolutional Network (GCN)** followed by a **GRU**. It looks at 12 previous reporting periods and predicts the next one.

```text
Information from 25 districts
            |
            v
   Graph Convolution (GCN)
   combine district information
            |
            v
   District representations
            |
            v
       GRU over 12 periods
       learn recent changes
            |
            v
        Linear output
            |
            v
     Next-period forecast
```

## 2.1 How the GCN part works

The baseline graph is a **queen-contiguity graph**. Two districts are connected when their geographical boundaries touch. It has 25 districts, 57 edges, no isolated districts, and an average degree of 4.56.

At each time step, a district combines its own information with information from connected districts. The graph operation is applied twice, so information can travel through more than one neighbouring district.

For example:

```text
District A <-- District B <-- District C
```

After two graph steps, information from C can influence A through B.

The graph is normalised so districts with many neighbours do not automatically have more influence.

## 2.2 How the GRU part works

After graph processing at each time step, the district information is passed through a shared GRU. The GRU looks at the recent 12-period sequence:

```text
t-11, t-10, ..., t-2, t-1, t  --->  GRU  --->  t+1 forecast
```

In simple terms, the GCN asks **“what information from other districts may be useful?”**, while the GRU asks **“how has this district's situation been changing recently?”**

## 2.3 Target used by the baseline

The model predicts the change between the current and next period on a log scale:

```text
log(1 + cases[t+1]) - log(1 + cases[t])
```

The output is converted back to cases. If the predicted change is zero, the forecast is essentially the current case count. This gives the model a persistence-like reference and makes it learn the correction rather than the entire case count from scratch.

The baseline has **8,193 trainable parameters**.

---

# 3. Baseline evaluation and Fold 9 results

The project uses nine chronological walk-forward folds with test years 2017–2025. Each test year is kept separate from training, and COVID years are treated as a separate distribution-shift analysis.

For this supervisor discussion, the direct comparison is **Fold 9 (2025)**.

## Fold 9 — 2025

| Model | MAE | RMSE | Peak MAE |
|---|---:|---:|---:|
| Persistence | **8.95** | — | — |
| GCN + GRU, v1 | **9.06** | **16.34** | **18.83** |
| Learnable-lag GCN + GRU | **9.14** | **16.47** | **19.46** |

### Why MAE is the main metric

MAE is easy to explain because it is measured in case-count units. An MAE of 9.06 means the average absolute forecasting error is about 9 cases per district-period. RMSE and Peak MAE are retained as supporting metrics because they give more information about larger errors and high-case periods.

### Fold 9 insight

The baseline GCN + GRU has an MAE of **9.06**, while persistence has **8.95**. This tells us that recent dengue case history is already a very strong signal for a one-period-ahead forecast in 2025.

This is important for the research: a new climate or graph feature has to add information beyond what recent case counts already provide.

The full nine-fold experiment also gives an important architectural observation: the GRU-only control performs better than GCN + GRU on all nine folds. This suggests that the current geographical graph is not necessarily the most useful way to describe dengue relationships. This directly motivates the planned data-driven graph rather than meaning that spatial information itself is useless.

---

# 4. Learnable climate lag encoder

## 4.1 Why introduce it?

Climate effects may not appear in dengue cases immediately. A simple solution is to manually include trailing climate averages, such as rainfall from 4, 8 and 12 periods ago. However, one fixed set of delays may not be appropriate for every district or every climate variable.

The new lag encoder asks:

> **Can the model learn which parts of the previous climate history are useful for each district and climate variable?**

## 4.2 Mechanism in simple terms

The encoder looks backwards over **26 reporting periods** for each climate variable.

```text
Current weather       0 periods ago
Previous weather      1 period ago
Previous weather      2 periods ago
...
Previous weather      25 periods ago
```

Instead of giving every delay an independent parameter, the encoder uses **six smooth Gaussian-shaped patterns** across the 26 periods. Each district has a small learned representation that determines how these patterns are combined for each climate variable.

In simple terms:

```text
Past climate history
        |
        v
Find useful delay pattern
        |
        v
Weighted climate history
        |
        v
Existing GCN + GRU
        |
        v
Forecast
```

The encoder has only **548 parameters**. It is placed before the existing GCN + GRU, while the main forecasting backbone is kept unchanged. Because it reaches 26 periods back and the backbone uses 12 periods, the input history becomes 37 periods.

## 4.3 Evidence that the mechanism works

The project contains a controlled synthetic recovery test where known delays of approximately 3–19 periods are planted. The encoder recovers those delays within about ±2 periods. This supports the implementation itself.

The independent lag analysis also finds delayed climate relationships in the real data. In Fold 9, rainfall shows district-level peaks around **5–10 periods**, with a median around 8 periods after deseasonalising. The relationship is not equally strong for every district; this is one reason a district-specific mechanism is useful to investigate.

---

# 5. Learnable lag results — Fold 9

| Model | MAE | RMSE | Peak MAE |
|---|---:|---:|---:|
| Baseline GCN + GRU | **9.06** | **16.34** | **18.83** |
| Learnable-lag GCN + GRU | **9.14** | **16.47** | **19.46** |

The learnable-lag model is **very close to the baseline** on Fold 9. The MAE difference is only about **0.08 cases**, RMSE differs by about **0.13**, and Peak MAE differs by about **0.63**.

### How to explain this to supervisors

The current result does not yet show a forecasting gain from the learned lags on this fold. However, the result is useful diagnostically.

There is a strong reason why a climate feature may have difficulty improving a one-period-ahead forecast: **the current dengue case count is already highly informative about the next case count**. Therefore, the additional climate signal can be relatively small compared with the case-history signal.

The separate lag analysis shows that delayed climate relationships exist, while the synthetic test shows that the encoder can recover known delays. The current result therefore suggests that the main challenge is not necessarily the ability to represent a delay; it may be that the one-step forecasting loss does not provide a strong enough signal for the encoder to learn the real-world delays reliably.

A good way to state the result is:

> **The learnable lag mechanism is functioning and the data contains delayed climate relationships, but those delays do not yet translate into a measurable improvement in one-period-ahead forecasting on Fold 9.**

This gives us a concrete direction for further experiments rather than changing the feature without understanding the result.

---

# 6. Planned graph feature: mapping districts with similar dengue behaviour

## 6.1 Motivation

The current graph is based only on physical boundaries. This is a useful starting point, but it assumes that sharing a border means two districts should exchange similar amounts of information.

The baseline experiments motivate questioning this assumption because the GRU-only model performs better than the GCN + GRU model across all nine folds.

The next question is:

> **Instead of defining district relationships from geography, can we define them from the dengue data itself?**

## 6.2 Proposed mechanism

### Step 1 — Build a historical case series for each district

Each district has a sequence of dengue cases over time.

```text
Colombo:  [c1, c2, c3, ..., cn]
Galle:    [g1, g2, g3, ..., gn]
Kandy:    [k1, k2, k3, ..., kn]
```

For each fold, the graph should be created using **training data only**. This prevents the future test year from influencing the graph.

### Step 2 — Measure similarity between districts

For every pair of districts, measure how similarly their case counts change over time. A simple first approach is correlation.

```text
Similarity(A, B)
    = correlation between A's and B's historical case patterns
```

If two districts often rise and fall at similar times, their similarity will be high.

### Step 3 — Keep the strongest relationships

Connecting every district to every other district would make the graph too dense. We can keep only the **top-k most similar districts** for each district, or keep connections above a chosen similarity threshold.

```text
             Gampaha
                |
                |
Colombo --------+-------- Kalutara
                |
               Galle
```

The exact k/threshold should be selected using validation data.

### Step 4 — Use similarity as the connection strength

Rather than simply using 0/1 connections, the similarity value can become the graph weight:

```text
High similarity   -> strong connection
Medium similarity -> weaker connection
Low similarity    -> no connection
```

This creates a **weighted data-driven graph**.

### Step 5 — Feed the new graph into the existing GCN + GRU

```text
Historical dengue cases
          |
          v
Calculate district similarities
          |
          v
Case-similarity graph
          |
          v
GCN using new graph
          |
          v
GRU over recent history
          |
          v
Next-period forecast
```

The key comparison should be:

```text
A. Identity / no graph
B. Current geographical graph
C. Case-similarity graph
D. Geographic + case-similarity graph
```

This experiment can tell us whether spatial information itself is useful, or whether the problem is mainly the way the spatial relationships are defined.

---

# 7. How the research components fit together

| Component | Main question | Approach |
|---|---|---|
| Baseline GCN + GRU | How do recent cases and districts affect the forecast? | Geographic graph + temporal history |
| Learnable lag encoder | When does climate information become useful? | Learn district/feature-specific climate delays |
| Case-similarity graph | Which districts behave similarly? | Build relationships from historical dengue cases |

Overall:

```text
                 Dengue + Climate History
                           |
              +------------+------------+
              |                         |
              v                         v
       Learnable climate          Case relationships
             delays                      |
              |                         v
              |                  Data-driven graph
              |                         |
              +------------+------------+
                           |
                           v
                    Spatial processing
                           |
                           v
                          GRU
                           |
                           v
                    Next-period forecast
```

### One-sentence research story

> **We are moving from manually fixed climate delays and manually defined geographical relationships toward learning when climate matters and which districts are actually related in dengue behaviour.**

---

# 8. Key points to tell supervisors

1. **The baseline is reproducible and lightweight:** GCN + GRU, 12-period lookback, 8,193 parameters.
2. **Fold 9 baseline MAE is 9.06**, with RMSE 16.34 and Peak MAE 18.83.
3. **Persistence is 8.95 on Fold 9**, showing how strong recent case history is for one-step forecasting.
4. **The current geographical graph is worth revisiting:** the full nine-fold experiment shows GRU-only outperforming GCN + GRU on every fold.
5. **The learnable lag encoder is close to the baseline on Fold 9:** MAE 9.14 versus 9.06, rather than showing a clear improvement yet.
6. **The lag mechanism has supporting evidence:** it recovers planted delays in controlled tests, and real-data analysis shows delayed climate relationships.
7. **The next graph experiment is directly motivated by the baseline:** replace the assumption that geographical neighbours are the most relevant neighbours with relationships learned from dengue case behaviour.
8. **All graph experiments should use training data only to construct the graph** and then use the same test folds and metrics for a fair comparison.

---

## Repository references

- `docs/baseline.md` — baseline architecture, evaluation and full results.
- `docs/model_tensors.md` — model inputs, folds, preprocessing and adjacency.
- `docs/learnable_lags.md` — learnable lag design.
- `docs/learnable_lags_results.md` — lag experiment results and interpretation.
- `scripts/16.train_gcn_gru.py` — baseline training.
- `scripts/18.train_lag_gcn_gru.py` — learnable-lag experiment.
- `src/models/lag_encoder.py` — lag encoder implementation.
