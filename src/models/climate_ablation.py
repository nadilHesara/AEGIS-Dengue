"""
Climate ablation: is climate what produces the h=3-4 skill?

`docs/multi_horizon.md` established the first real improvement in this project.
At h=3 and h=4 the model beats same-horizon persistence on MAE *and* peak MAE,
7/7 folds at h=4, p < 0.01, and the gain is not the 2017 artefact. The
explanation offered for it is mechanistic: the measured rainfall-to-dengue delay
is 5-10 weeks (`scripts/17`), so as the horizon grows the forecast origin's grip
weakens and climate gets room to matter.

**That explanation is currently untested.** Skill appearing at h=4 is equally
consistent with a duller story: persistence degrades faster than the model does,
so the *relative* number improves while climate contributes nothing. The two
readings make opposite predictions about what happens when climate is removed,
and this module is the instrument that separates them.

Three arms:

    full        every channel. The h=8f model.

    no_climate  the climate channels sliced out. At v1 that is 16 of 23 --
                the 7 instantaneous channels and the 9 rolling lags -- leaving
                case history, seasonality, the observation flags and the
                centroids. Sliced, not zeroed: a zeroed channel still costs the
                first graph convolution its weights and still passes through the
                per-fold scaler, so it is not the same experiment
                (`scripts/18.FeatureSubsetGCNGRU` makes the same choice for the
                same reason).

    shuffled    every channel present, but the climate ones time-shuffled within
                each district. This is the arm that makes the result mean
                something.

**Why the shuffled arm is not optional.** `docs/learnable_lags_results.md` 6.3
ran the two-arm version of this at h=1 and flagged its own result as confounded:
fold 1 is both the epidemic year and the smallest fold, so extra input channels
may be helping a data-starved model regardless of what those channels contain.
Dropping 16 of 23 channels changes the input width, the first layer's parameter
count and the effective regularisation all at once. If `no_climate` scores worse,
that confound means it is not yet evidence that *climate* was the thing helping.

The shuffled arm holds all of it fixed. Same width, same parameter count, same
per-fold scaler, same marginal distribution per channel per district -- only the
alignment between climate and time is destroyed. So:

    full ~ shuffled           climate's *content* carries nothing; any gap to
                              `no_climate` is width, not weather.

    full < shuffled ~ no_climate   climate carries real temporal signal. This is
                              the reading that supports the mechanism claim, and
                              it is the one the horizon trend should sharpen.

**The shuffle is within-district and within-split.** Shuffling across districts
would leak one district's weather into another's window and destroy the spatial
structure as well, which is a second change and would confound the confound
control. Shuffling across the train/test boundary would leak test-period weather
into training. Both are avoided: each district's windows are permuted only among
themselves, inside a single split, under a seeded generator.

**The target is never touched.** Only the climate channels of `X` move. Case
history, seasonality, flags, centroids, `y`, the mask and the anchor are all
left exactly as the other arms see them.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


# The instantaneous climate channels, as named by `scripts/12.build_model_tensors`.
# The same seven `scripts/18` smooths, and for the same reason: case history,
# seasonality, the observation flags and the centroids are not weather.
CLIMATE_FEATURES = (
    "rainfall_daily_mean_mm",
    "rainy_days_frac",
    "temperature_mean_c",
    "diurnal_range_c",
    "dewpoint_mean_c",
    "relative_humidity_mean",
    "wind_speed_mean",
)

# v1 adds rolling means of three of them at 4, 8 and 12 periods. These are the
# channels that carry *delayed* climate, so they are the ones the 5-10 week
# delay story actually runs through: dropping the instantaneous channels while
# keeping `rainfall_..._roll12` would leave the mechanism intact and call it an
# ablation. Matched by suffix so a future rolling window is caught automatically.
ROLLING_SUFFIXES = ("_roll4", "_roll8", "_roll12")


def climate_indices(feature_names: list[str]) -> list[int]:
    """Return the indices of every climate-derived channel.

    Both the instantaneous channels and any rolling transform of one. A channel
    counts as climate if it is in `CLIMATE_FEATURES`, or if stripping a rolling
    suffix leaves a name that is.
    """

    indices = []

    for index, name in enumerate(feature_names):
        base = name
        for suffix in ROLLING_SUFFIXES:
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                break

        if base in CLIMATE_FEATURES:
            indices.append(index)

    return indices


def non_climate_indices(feature_names: list[str]) -> list[int]:
    """Return the indices of every channel that is not climate-derived."""

    dropped = set(climate_indices(feature_names))

    return [index for index in range(len(feature_names)) if index not in dropped]


class FeatureSubsetGCNGRU(nn.Module):
    """The baseline backbone, shown only some of the input channels.

    Slices the feature axis before the backbone sees it, so the backbone is
    genuinely built for the narrower input rather than being handed zeros. The
    caller sizes the backbone with `len(keep_indices)`.

    Mirrors `scripts/18.FeatureSubsetGCNGRU`, reimplemented here so this
    experiment does not import a private class out of a numbered script.
    """

    def __init__(self, backbone: nn.Module, keep_indices: list[int]):
        super().__init__()

        self.backbone = backbone
        self.register_buffer(
            "keep_indices", torch.tensor(keep_indices, dtype=torch.long)
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.backbone(x[..., self.keep_indices], adjacency)


def shuffle_climate(
    x: np.ndarray,
    indices: list[int],
    seed: int,
) -> np.ndarray:
    """Time-shuffle the climate channels within each district.

    Parameters
    ----------
    x:
        Windowed inputs, `[windows, steps, nodes, features]`. Not modified; a
        copy is returned.
    indices:
        Which feature channels to shuffle. Everything else is passed through
        untouched.
    seed:
        Generator seed, so an arm is reproducible.

    The permutation is over the **window** axis, drawn independently per
    district and per channel, and applied to the whole `[steps]` block of a
    window at once. That is the shuffle that preserves what needs preserving:

      - the marginal distribution of each channel, per district, exactly -- the
        same values appear, in a different order, so the per-fold scaler sees
        identical statistics;
      - the internal structure of each 12-step window, so a window still looks
        like a plausible stretch of weather rather than noise;

    while destroying the only thing the mechanism claim depends on: which
    stretch of weather sits in front of which outbreak.

    Permuting each district and channel independently is deliberate. A single
    shared permutation would preserve the cross-district and cross-channel
    correlation structure of the weather, leaving a model able to exploit "it
    rained everywhere at once" as a seasonal proxy. Independent permutations
    remove that too, which makes this the conservative control: it destroys
    strictly more, so `full ~ shuffled` is strong evidence of no content.
    """

    if not indices:
        return x.copy()

    shuffled = x.copy()
    generator = np.random.default_rng(seed)

    n_windows = x.shape[0]
    n_nodes = x.shape[2]

    for node in range(n_nodes):
        for channel in indices:
            order = generator.permutation(n_windows)
            shuffled[:, :, node, channel] = shuffled[order, :, node, channel]

    return shuffled


def shuffle_split_climate(
    arrays: dict[str, dict[str, np.ndarray]],
    indices: list[int],
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Apply `shuffle_climate` to every split, independently.

    Each split is permuted only among its own windows, so no weather crosses the
    train/validation/test boundary -- shuffling the pooled array would move test-
    period weather into training windows and leak.

    Only `X` changes. `y`, the mask and the anchor are passed through by
    reference, so the target the model is scored against is identical to what
    every other arm sees.
    """

    out: dict[str, dict[str, np.ndarray]] = {}

    for offset, (split, content) in enumerate(arrays.items()):
        shuffled = dict(content)
        shuffled["X"] = shuffle_climate(content["X"], indices, seed + 1000 * offset)
        out[split] = shuffled

    return out
