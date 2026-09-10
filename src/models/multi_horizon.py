"""
Multi-horizon forecasting for the GCN+GRU: separate heads, one shared trunk.

Every result in this repository so far is at horizon 1, and that single fact is
the standing explanation for most of the negative ones. `docs/learnable_lags_results.md`
measured it directly: at one week ahead the previous period's case count carries
nearly all the forecasting signal, and dropping every climate channel costs
+0.01 MAE. That is why the learnable lag encoder found no gradient (§7), why the
reweighted objective only bites in epidemic conditions (§8), why the activation
sweep was flat (§8b), why the hyperparameter surface is flat (§8c), and why no
graph beats no graph (§8e).

If that explanation is right, then extending the horizon should change the
picture, because the origin's grip on the target has to weaken as the gap grows.
The measured rainfall-to-dengue delay is 5-10 weeks (`scripts/17`), so at h=4
climate is closer to being load-bearing than at h=1. If the explanation is
wrong, longer horizons will simply be uniformly harder and nothing else will
change. Either way the answer is informative.

Two things live here.

`make_multi_horizon_windows` cuts windows carrying **all** horizons at once.
`scripts/12.make_windows` emits one target per window, at `t + horizon`; a
shared model needs `y[t+1] .. y[t+H]` for the same input window, each with its
own mask. The alignment rule is unchanged and is the one that matters: inputs
run `X[t - lookback + 1 : t + 1]`, every target is strictly after `t`, and the
window count is set by the **longest** horizon so that every horizon in a given
run is scored on exactly the same origins. That last point is what makes the
per-horizon numbers comparable to each other rather than each being measured on
a slightly different set of weeks.

`MultiHorizonGCNGRU` wraps a plain `GCNGRU` and replaces its single-output head
with one linear head per horizon over the **same** GRU state. The trunk -- graph
convolution, shared GRU, dropout -- is untouched and shared; only the heads are
per-horizon. That is the design the comparison is about:

    separate runs   H independent models, each with its own trunk, each trained
                    on its own horizon. H times the parameters and H times the
                    training cost. Nothing is shared, so nothing can transfer.

    shared model    one trunk, H heads, trained on the summed masked loss. The
                    trunk has to learn a representation useful for all H
                    horizons at once. Cheaper, and the near-horizon targets act
                    as auxiliary supervision for the far ones -- which is the
                    mechanism by which a shared model can beat separate ones
                    despite having less capacity per horizon.

The head is deliberately the only per-horizon part. Giving each horizon its own
GRU would make the "shared" model H separate models in a trench coat, and the
comparison against separate runs would measure nothing.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


DEFAULT_HORIZONS = (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def make_multi_horizon_windows(
    tensors: dict[str, np.ndarray],
    lookback: int,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    drop_incomplete: bool = False,
) -> dict[str, np.ndarray]:
    """Cut windows carrying every horizon's target for the same input window.

    Returns `y` and `y_mask` shaped [windows, nodes, horizons] rather than
    [windows, nodes]; `target_period_id` is [windows, horizons]. `X`,
    `origin_period_id` and `input_period_id` keep the shapes
    `scripts/12.make_windows` gives them.

    The number of windows is set by `max(horizons)`, so every horizon is scored
    on the same forecast origins. Trimming per horizon instead would leave the
    h=1 column measured on windows the h=4 column never saw, and the columns
    would not be comparable.

    A window is kept when *any* district has an observed target at *any*
    horizon; individual missing cells stay masked, exactly as the single-horizon
    windower does per node. Dropping a window because one horizon-district cell
    is absent would discard the other 99 observed cells.
    """

    if lookback < 1:
        raise ValueError("lookback must be at least 1.")

    if not horizons:
        raise ValueError("horizons must not be empty.")

    if any(h < 1 for h in horizons):
        raise ValueError("every horizon must be at least 1.")

    x = tensors["X"]
    y = tensors["y"]
    y_mask = tensors["y_mask"]
    period_id = tensors["period_id"]

    longest = max(horizons)
    n_periods = x.shape[0]
    origins = np.arange(lookback - 1, n_periods - longest)

    if len(origins) == 0:
        raise ValueError(
            f"lookback {lookback} and horizon {longest} leave no windows in "
            f"{n_periods} periods."
        )

    offsets = np.arange(-lookback + 1, 1)
    window_index = origins[:, None] + offsets[None, :]

    x_windows = x[window_index]

    # [windows, nodes, horizons] -- one target column per horizon, same origins.
    y_windows = np.stack([y[origins + h] for h in horizons], axis=-1)
    mask_windows = np.stack([y_mask[origins + h] for h in horizons], axis=-1)
    target_periods = np.stack([period_id[origins + h] for h in horizons], axis=-1)

    keep = (mask_windows == 1).any(axis=(1, 2))
    if drop_incomplete:
        keep &= ~np.isnan(x_windows).any(axis=(1, 2, 3))

    return {
        "X": x_windows[keep],
        "y": y_windows[keep],
        "y_mask": mask_windows[keep],
        "origin_period_id": period_id[origins][keep],
        "target_period_id": target_periods[keep],
        "input_period_id": period_id[window_index][keep],
        "horizons": np.asarray(horizons, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiHorizonGCNGRU(nn.Module):
    """A `GCNGRU` with one linear head per horizon over a shared GRU state.

    The trunk is the baseline's, untouched: graph convolution over the districts
    at every step, then one shared GRU over the window, then dropout. Only the
    output head is replicated, once per horizon.

    Sharing the trunk is the whole point. The alternative -- a separate GRU per
    horizon -- would be H independent models sharing a constructor, and
    comparing that against H separate training runs would measure nothing. Here
    the representation is forced to serve all horizons at once, so the near
    horizons act as auxiliary supervision for the far ones.

    Returns [batch, nodes, horizons], with column `i` the forecast for
    `horizons[i]`.
    """

    def __init__(self, backbone: nn.Module, n_horizons: int):
        super().__init__()

        if n_horizons < 1:
            raise ValueError("n_horizons must be at least 1.")

        self.backbone = backbone
        self.n_horizons = n_horizons

        hidden = backbone.gru.hidden_size
        self.heads = nn.ModuleList(nn.Linear(hidden, 1) for _ in range(n_horizons))

        # The backbone's own head is bypassed: this module's heads read the GRU
        # state directly. Deleting it keeps the parameter count honest -- an
        # unused head would still be reported as trainable.
        self.backbone.head = nn.Identity()

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, _ = x.shape

        spatial = x
        for layer in self.backbone.graph_layers:
            spatial = torch.relu(layer(spatial, adjacency))
            spatial = self.backbone.dropout(spatial)

        sequences = spatial.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)

        output, _ = self.backbone.gru(sequences)
        last = self.backbone.dropout(output[:, -1])

        # One head per horizon over the same state, concatenated on the last axis.
        predictions = [head(last) for head in self.heads]

        return torch.cat(predictions, dim=-1).view(batch, nodes, self.n_horizons)


def masked_multi_horizon_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked MSE summed over horizons, normalised by the observed cell count.

    Every horizon contributes to one scalar loss. The denominator is the total
    number of observed cells across all horizons rather than per horizon, so the
    loss stays on the same scale as the baseline's single-horizon `masked_mse`
    and the early-stopping patience does not have to be retuned.

    Horizons are weighted equally. An alternative would be to downweight the far
    ones because they are harder, but that would build a preference into the
    objective that nothing has measured; equal weighting is the choice that
    assumes least.
    """

    error = (prediction - target) * mask
    denominator = mask.sum().clamp(min=1.0)

    return (error ** 2).sum() / denominator
