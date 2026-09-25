"""
Build stage 3 model tensors from the master weekly panel.

Stage 3 is the sequence stage named in `docs/climate_dataset_schema.md`: the
point where the flat panel becomes arrays a sequence model can index. Two
things happen here and nothing else.

    The panel is reshaped into a dense [period, node, feature] cube. Row order
    is period_id then node_id, so X[t, i] and A[i, j] refer to the same
    district by construction. node_id 0-24 comes from nodes.csv and is a
    permanent part of the model definition.

    Trailing rolling summaries of climate are added in the v1 variant. They
    look backwards only and include the current period, which is observable at
    the forecast origin. Nothing here looks forward.

Three feature variants are emitted:

    v0  raw per-period features. The recurrent layer has to discover any lag
        structure by itself over the input window.

    v1  v0 plus trailing 4, 8 and 12 period means of rainfall, temperature and
        humidity, which is where the mosquito development lag is expected to
        sit.

    v2  v1 plus two outbreak-history columns computed from the case series
        alone: `national_wave_rank` (this district's case count ranked
        against all 25 districts in the same period) and
        `trailing_52_cumulative_cases` (log1p of this district's own cases
        summed over the trailing 52 periods, inclusive). Both are
        period-local or backward-looking only -- see `add_history_features`
        -- so they carry no information a real forecast origin would not
        already have. The residual diagnostic
        (`scripts/evaluation/30.residual_diagnostic.py`) found both proxies correlate
        with the baseline model's test-set error more strongly than any
        climate channel, which is the motivation for building them as real
        inputs rather than leaving them as a diagnostic-only computation.

v1 minus v0 measures what hand-specified lags are worth on this data. That
difference is the number a learnable lag module has to beat, so it is built
now rather than reconstructed later.

Missing values are preserved as NaN. No scaling, no imputation and no
train/validation/test split happens here: all three must be fitted on the
training split alone and therefore belong to the training script, not to a
frozen artefact.

Outputs:
    data/processed/model_tensors_v0.npz
    data/processed/model_tensors_v1.npz
    data/processed/model_tensors_v2.npz
    results/models/model_tensors_report.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]

PANEL_PATH = PROJECT_DIR / "data" / "processed" / "panel_weekly.parquet"
NODES_PATH = PROJECT_DIR / "data" / "processed" / "nodes.csv"

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "models"

REPORT_PATH = RESULTS_DIR / "model_tensors_report.md"

TENSOR_VERSION = "tensors-v1"

EXPECTED_NODES = 25

# Trailing windows, in reporting periods, for the v1 lag summaries.
ROLLING_WINDOWS = (4, 8, 12)

# Climate variables given trailing means in v1.
ROLLING_SOURCES = (
    "rainfall_daily_mean_mm",
    "temperature_mean_c",
    "relative_humidity_mean",
)

# Trailing window, in reporting periods, for v2's cumulative-case history
# column. Matches scripts/evaluation/30.residual_diagnostic.py's DEPLETION_WINDOW.
HISTORY_WINDOW = 52

NATIONAL_WAVE_RANK = "national_wave_rank"
TRAILING_CUMULATIVE_CASES = "trailing_52_cumulative_cases"

NEIGHBOR_CASES_MEAN = "neighbor_cases_log1p_mean"
NEIGHBOR_CASES_MAX = "neighbor_cases_log1p_max"
NEIGHBOR_CASE_VELOCITY = "neighbor_case_velocity"
NEIGHBOR_FEATURES = [NEIGHBOR_CASES_MEAN, NEIGHBOR_CASES_MAX, NEIGHBOR_CASE_VELOCITY]

# Days in a mean tropical year, used for the seasonal angle. Not 365: the span
# contains five leap years and day-of-year drifts against the season without it.
DAYS_PER_YEAR = 365.25

PANEL_COLUMNS = [
    "period_id",
    "node_id",
    "start_date",
    "reporting_days",
    "cases",
    "case_observed",
    "rainfall_daily_mean_mm",
    "rainy_days",
    "temperature_mean_c",
    "temperature_min_c",
    "temperature_max_c",
    "dewpoint_mean_c",
    "relative_humidity_mean",
    "wind_speed_mean",
    "weather_complete",
]

NODE_COLUMNS = ["node_id", "canonical_name", "centroid_lat", "centroid_lon"]

BASE_FEATURES = [
    "cases_log1p",
    "rainfall_daily_mean_mm",
    "rainy_days_frac",
    "temperature_mean_c",
    "diurnal_range_c",
    "dewpoint_mean_c",
    "relative_humidity_mean",
    "wind_speed_mean",
    "doy_sin",
    "doy_cos",
    "weather_observed",
    "case_observed",
    "centroid_lat",
    "centroid_lon",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_panel(path: Path = PANEL_PATH) -> pd.DataFrame:
    """Load the master weekly panel, ordered period_id then node_id."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/9.aggregate_climate_to_periods.py "
            f"and scripts/10.create_master_panel.py first."
        )

    panel = pd.read_parquet(path)[PANEL_COLUMNS].copy()
    panel["start_date"] = pd.to_datetime(panel["start_date"])

    return panel.sort_values(["period_id", "node_id"]).reset_index(drop=True)


def load_nodes(path: Path = NODES_PATH) -> pd.DataFrame:
    """Load the canonical district registry."""

    return pd.read_csv(path)[NODE_COLUMNS].sort_values("node_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Grid checks
# ---------------------------------------------------------------------------

def check_registry(nodes: pd.DataFrame) -> None:
    """Assert the node registry is the canonical 25 districts."""

    if len(nodes) != EXPECTED_NODES:
        raise ValueError(f"nodes.csv has {len(nodes)} rows, expected {EXPECTED_NODES}.")

    expected = np.arange(EXPECTED_NODES)
    if not np.array_equal(nodes["node_id"].to_numpy(), expected):
        raise ValueError("node_id must be the dense sequence 0-24.")


def check_grid(panel: pd.DataFrame, nodes: pd.DataFrame) -> None:
    """Assert the panel is a complete, correctly ordered period x node grid."""

    periods = panel["period_id"].unique()
    node_ids = panel["node_id"].unique()

    if not np.array_equal(np.sort(node_ids), nodes["node_id"].to_numpy()):
        raise ValueError("Panel node_id values do not match nodes.csv.")

    if len(panel) != len(periods) * len(node_ids):
        raise ValueError(
            f"Panel has {len(panel)} rows, expected {len(periods) * len(node_ids)} "
            f"for a complete grid."
        )

    # period_id is a dense integer sequence by construction of the calendar.
    # Sequence windows index it directly, so a gap here would silently shorten
    # the lag reach of every window that crossed it.
    expected_periods = np.arange(periods.min(), periods.max() + 1)
    if not np.array_equal(np.sort(periods), expected_periods):
        raise ValueError("period_id is not a dense gap-free sequence.")

    counts = panel.groupby("period_id").size()
    if not (counts == len(node_ids)).all():
        offenders = counts[counts != len(node_ids)].index.tolist()
        raise ValueError(f"Periods without {len(node_ids)} rows: {offenders}")

    # The reshape below relies on this ordering and cannot detect a violation.
    tiled = np.tile(nodes["node_id"].to_numpy(), len(periods))
    if not np.array_equal(panel["node_id"].to_numpy(), tiled):
        raise ValueError("Panel rows are not ordered period_id then node_id.")


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def mask_unobserved_weather(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the panel with fabricated weather values restored to NaN.

    The aggregation emits rainy_days as an integer count, so a period with no
    weather at all arrives as rainy_days = 0. That is a fabricated dry period,
    not an observation, and the schema requires absence to be carried as
    absence. Every other weather column is already NaN on those rows.
    """

    panel = panel.copy()
    unobserved = ~panel["weather_complete"].astype(bool)
    panel.loc[unobserved, "rainy_days"] = np.nan

    return panel


def add_base_features(panel: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Attach the v0 feature columns to the panel."""

    panel = panel.merge(
        nodes[["node_id", "centroid_lat", "centroid_lon"]],
        on="node_id",
        how="left",
        validate="many_to_one",
    )

    # log1p because cases span 0 to 2631 with a median of 10. Trained on raw
    # counts the loss is decided entirely by Colombo's peaks.
    panel["cases_log1p"] = np.log1p(panel["cases"])

    # Normalised by reporting_days, not assumed 7: periods 122 and 127 are 8
    # and 6 days long, so a raw count is not comparable across the series.
    panel["rainy_days_frac"] = panel["rainy_days"] / panel["reporting_days"]

    panel["diurnal_range_c"] = panel["temperature_max_c"] - panel["temperature_min_c"]

    # Seasonality from the period start date. source_week is unusable here:
    # the interval labelled 2026 week 53 falls in December 2025.
    angle = 2 * np.pi * panel["start_date"].dt.dayofyear / DAYS_PER_YEAR
    panel["doy_sin"] = np.sin(angle)
    panel["doy_cos"] = np.cos(angle)

    panel["weather_observed"] = panel["weather_complete"].astype(float)
    panel["case_observed"] = panel["case_observed"].astype(float)

    return panel


def rolling_feature_name(source: str, window: int) -> str:
    """Return the column name for a trailing mean of one climate variable."""

    return f"{source}_roll{window}"


def add_rolling_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Attach trailing per-node rolling means and return their column names.

    The window is right-closed and includes the current period, which is
    observable at the forecast origin. min_periods equals the window length, so
    a partial mean is never emitted: the leading periods are NaN and the
    windows that touch them are dropped rather than trained on a mean of three
    periods labelled as a mean of twelve.
    """

    panel = panel.copy()
    names: list[str] = []

    for source in ROLLING_SOURCES:
        for window in ROLLING_WINDOWS:
            name = rolling_feature_name(source, window)
            # Rows are ordered period_id then node_id, so within one node group
            # they are already in chronological order.
            panel[name] = panel.groupby("node_id")[source].transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
            names.append(name)

    return panel, names


def add_history_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Attach the two outbreak-history columns and return their names.

    Both are computed causally: a district-period's value uses only that
    period's own row (the rank) or that district's own history at or before
    that period (the cumulative sum). Neither reads anything from a later
    period. The origin/target split that actually enforces causality at
    inference time is `make_windows`'s -- these columns sit in `X` like every
    other feature, and the model only ever sees `X` up to its own forecast
    origin.

        national_wave_rank              rank of this district's case count
                                         among all 25 districts in the SAME
                                         period, 1 = highest. Period-local:
                                         no other period is touched, so this
                                         needs no fold-aware fitting the way
                                         scripts/30's percentile threshold
                                         does for `periods_since_outbreak`.

        trailing_52_cumulative_cases    log1p of this district's own cases
                                         summed over the trailing 52 periods,
                                         inclusive of the current one.
                                         `min_periods=52`, so the first 51
                                         periods of the series are NaN rather
                                         than a partial sum mislabeled as a
                                         full one -- the same convention
                                         `add_rolling_features` uses for the
                                         climate lag means, and for the same
                                         reason. A NaN case count anywhere
                                         inside the trailing window propagates
                                         to NaN rather than being treated as
                                         zero cases observed: an imputed count
                                         is not an observation, and pretending
                                         otherwise here would be exactly the
                                         fabricated-zero failure
                                         `mask_unobserved_weather` exists to
                                         prevent for `rainy_days`.

                                         `log1p` is applied to the summed
                                         count, not to `cases` before summing
                                         -- the sum itself stays linear, only
                                         its scale is compressed afterwards.
                                         Uncompressed, this column ranges into
                                         the tens of thousands during 2017 and
                                         z-scores to double digits of standard
                                         deviation, which destabilised
                                         training on the identity backbone.
                                         `cases_log1p` already compresses raw
                                         counts for the same reason; this is
                                         the same fix applied to a sum instead
                                         of a single period's count. `log1p`
                                         propagates NaN unchanged, so the
                                         warm-up and missing-period NaN
                                         behaviour above is untouched by this.
    """

    panel = panel.copy()

    panel[NATIONAL_WAVE_RANK] = panel.groupby("period_id")["cases"].rank(
        ascending=False, method="min"
    )

    # Rows are ordered period_id then node_id, so within one node group they
    # are already in chronological order; `rolling` reads that order directly.
    panel[TRAILING_CUMULATIVE_CASES] = np.log1p(
        panel.groupby("node_id")["cases"].transform(
            lambda series: series.rolling(HISTORY_WINDOW, min_periods=HISTORY_WINDOW).sum()
        )
    )

    return panel, [NATIONAL_WAVE_RANK, TRAILING_CUMULATIVE_CASES]


def add_neighbor_features(
    panel: pd.DataFrame,
    adjacency_binary: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    """Add causal spatial neighbor features using queen contiguity.

    Computes for each district i and period t:
    1. neighbor_cases_log1p_mean: Mean log1p(cases) of adjacent contiguous districts.
    2. neighbor_cases_log1p_max: Maximum log1p(cases) among adjacent contiguous districts.
    3. neighbor_case_velocity: Week-over-week change in neighbor mean case count.

    These are strictly causal: they only access counts at period t and t-1,
    providing regional wave awareness without blurring local signals.
    """
    panel = panel.copy()
    c = panel.pivot(index="period_id", columns="node_id", values="cases").to_numpy()
    log_c = np.log1p(c)
    n_nodes = adjacency_binary.shape[0]

    n_mean = np.stack(
        [np.nanmean(log_c[:, np.where(adjacency_binary[i] == 1)[0]], axis=1) for i in range(n_nodes)],
        axis=1,
    )
    n_max = np.stack(
        [np.nanmax(log_c[:, np.where(adjacency_binary[i] == 1)[0]], axis=1) for i in range(n_nodes)],
        axis=1,
    )
    n_vel = np.diff(n_mean, axis=0, prepend=np.nan)

    panel[NEIGHBOR_CASES_MEAN] = n_mean.ravel()
    panel[NEIGHBOR_CASES_MAX] = n_max.ravel()
    panel[NEIGHBOR_CASE_VELOCITY] = n_vel.ravel()

    return panel, list(NEIGHBOR_FEATURES)


def feature_names_for(
    variant: str,
    rolling_names: list[str],
    history_names: list[str] | None = None,
) -> list[str]:
    """Return the ordered feature list for one variant.

    `history_names` defaults to `None` (treated as empty) so callers built
    before v2 existed -- asking only for "v0" or "v1" -- keep working
    unchanged; only a "v2" or "v3" request needs it supplied.
    """

    if variant == "v0":
        return list(BASE_FEATURES)

    if variant == "v1":
        return list(BASE_FEATURES) + rolling_names

    if variant == "v2":
        if not history_names:
            raise ValueError("variant 'v2' requires history_names.")
        return list(BASE_FEATURES) + rolling_names + list(history_names)

    if variant == "v3":
        if not history_names:
            raise ValueError("variant 'v3' requires history_names.")
        return list(BASE_FEATURES) + rolling_names + list(history_names) + list(NEIGHBOR_FEATURES)

    raise ValueError(f"Unknown variant {variant!r}.")


# ---------------------------------------------------------------------------
# Reshaping
# ---------------------------------------------------------------------------

def build_tensors(
    panel: pd.DataFrame,
    features: list[str],
) -> dict[str, np.ndarray]:
    """Reshape the ordered panel into [period, node, feature] arrays."""

    periods = np.sort(panel["period_id"].unique())
    node_ids = np.sort(panel["node_id"].unique())

    n_periods = len(periods)
    n_nodes = len(node_ids)

    x = panel[features].to_numpy(dtype=np.float32).reshape(n_periods, n_nodes, -1)
    y = panel["cases"].to_numpy(dtype=np.float32).reshape(n_periods, n_nodes)

    y_mask = (
        panel["case_observed"].to_numpy(dtype=np.int8).reshape(n_periods, n_nodes)
    )
    weather_mask = (
        panel["weather_observed"].to_numpy(dtype=np.int8).reshape(n_periods, n_nodes)
    )

    start_dates = (
        panel.loc[panel["node_id"] == node_ids[0], "start_date"]
        .to_numpy()
        .astype("datetime64[D]")
    )

    return {
        "X": x,
        "y": y,
        "y_mask": y_mask,
        "weather_mask": weather_mask,
        "period_id": periods.astype(np.int32),
        "node_id": node_ids.astype(np.int32),
        "start_date": start_dates,
        "feature_names": np.array(features, dtype=object),
    }


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def make_windows(
    tensors: dict[str, np.ndarray],
    lookback: int,
    horizon: int = 1,
    drop_incomplete: bool = True,
) -> dict[str, np.ndarray]:
    """Cut sequence windows from the tensors for one lookback and horizon.

    A window with forecast origin t uses inputs X[t - lookback + 1 : t + 1] and
    predicts y[t + horizon]. Every input period is therefore at or before the
    origin, and the target is strictly after it. This is the only place the
    forecast alignment is defined; getting it wrong here is invisible
    downstream, which is why the test asserts it directly against period_id.

    Target missingness is handled per node, not per window. A window is dropped
    only when no district in it has an observed target; a window where one
    district is missing is kept and y_mask carries the gap, so the loss can
    exclude that one cell. Dropping the whole window instead would throw away 24
    observed districts to remove one absent Puttalam record.

    With drop_incomplete, windows holding any NaN input are also removed. That
    is the pre-imputation path: once scripts/14 has filled the features for a
    fold there are no NaN inputs left, and the flag can be turned off.
    """

    if lookback < 1:
        raise ValueError("lookback must be at least 1.")

    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    x = tensors["X"]
    y = tensors["y"]
    y_mask = tensors["y_mask"]
    period_id = tensors["period_id"]

    n_periods = x.shape[0]
    origins = np.arange(lookback - 1, n_periods - horizon)

    if len(origins) == 0:
        raise ValueError(
            f"lookback {lookback} and horizon {horizon} leave no windows in "
            f"{n_periods} periods."
        )

    offsets = np.arange(-lookback + 1, 1)
    window_index = origins[:, None] + offsets[None, :]

    x_windows = x[window_index]
    y_windows = y[origins + horizon]
    mask_windows = y_mask[origins + horizon]

    # A target period with nothing observed anywhere carries no signal at all.
    keep = (mask_windows == 1).any(axis=1)
    if drop_incomplete:
        keep &= ~np.isnan(x_windows).any(axis=(1, 2, 3))

    return {
        "X": x_windows[keep],
        "y": y_windows[keep],
        "y_mask": mask_windows[keep],
        "origin_period_id": period_id[origins][keep],
        "target_period_id": period_id[origins + horizon][keep],
        "input_period_id": period_id[window_index][keep],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarise_variant(
    tensors: dict[str, np.ndarray],
    lookback: int = 12,
    horizon: int = 1,
) -> dict[str, object]:
    """Return per-variant counts for the report."""

    x = tensors["X"]
    y_mask = tensors["y_mask"]
    period_id = tensors["period_id"]

    windows = make_windows(tensors, lookback=lookback, horizon=horizon)

    nan_by_feature = {
        name: int(np.isnan(x[:, :, index]).sum())
        for index, name in enumerate(tensors["feature_names"])
    }

    # Which windows were lost, and to what. A bare usable count hides the fact
    # that one missing district-period removes every window that reads it.
    # Counted here from the index arithmetic rather than from make_windows,
    # which has already applied its own filtering.
    origins = np.arange(lookback - 1, x.shape[0] - horizon)
    window_index = origins[:, None] + np.arange(-lookback + 1, 1)[None, :]

    dropped_nan = np.isnan(x[window_index]).any(axis=(1, 2, 3))
    dropped_target = (y_mask[origins + horizon] != 1).all(axis=1)
    dropped = dropped_nan | dropped_target

    # Targets kept, but with at least one district masked out of the loss.
    partial = ~dropped & (y_mask[origins + horizon] != 1).any(axis=1)

    return {
        "shape": x.shape,
        "n_features": x.shape[2],
        "nan_cells": int(np.isnan(x).sum()),
        "nan_by_feature": {k: v for k, v in nan_by_feature.items() if v > 0},
        "windows_total": len(origins),
        "windows_usable": len(windows["target_period_id"]),
        "dropped_nan_input": int(dropped_nan.sum()),
        "dropped_unobserved_target": int(dropped_target.sum()),
        "partial_target_windows": int(partial.sum()),
        "dropped_target_periods": period_id[origins + horizon][dropped].tolist(),
        "first_target_period": int(windows["target_period_id"].min()),
        "last_target_period": int(windows["target_period_id"].max()),
    }


def format_period_ranges(periods: list[int]) -> str:
    """Collapse a sorted period_id list into readable contiguous ranges."""

    if not periods:
        return "none"

    ranges: list[str] = []
    start = previous = periods[0]

    for period in periods[1:]:
        if period == previous + 1:
            previous = period
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = period

    ranges.append(str(start) if start == previous else f"{start}-{previous}")

    return ", ".join(ranges)


def write_report(
    summaries: dict[str, dict[str, object]], tensors_v2: dict, history_names: list[str]
) -> None:
    """Write the human-readable build report."""

    lines = [
        "# Model tensors",
        "",
        f"Version: `{TENSOR_VERSION}`",
        "",
        "Stage 3 arrays cut from `data/processed/panel_weekly.parquet`.",
        "Values are unscaled and unimputed: NaN is preserved so that scaling",
        "and imputation can be fitted on the training split alone.",
        "",
        "## Shapes",
        "",
        "| Variant | X | Features | Windows (L=12, h=1) |",
        "| --- | --- | --- | --- |",
    ]

    for variant, summary in summaries.items():
        shape = "x".join(str(dim) for dim in summary["shape"])
        lines.append(
            f"| `{variant}` | {shape} | {summary['n_features']} | "
            f"{summary['windows_usable']} of {summary['windows_total']} |"
        )

    lines += [
        "",
        "## Features",
        "",
        "| # | Name | Variant |",
        "| --- | --- | --- |",
    ]

    for index, name in enumerate(tensors_v2["feature_names"]):
        if name in BASE_FEATURES:
            variant = "v0, v1, v2"
        elif name in history_names:
            variant = "v2"
        else:
            variant = "v1, v2"
        lines.append(f"| {index} | `{name}` | {variant} |")

    lines += [
        "",
        "## Missing cells",
        "",
    ]

    for variant, summary in summaries.items():
        lines.append(f"### `{variant}`")
        lines.append("")
        lines.append(f"Total NaN cells: {summary['nan_cells']}")
        lines.append("")

        if summary["nan_by_feature"]:
            lines.append("| Feature | NaN cells |")
            lines.append("| --- | --- |")
            for name, count in summary["nan_by_feature"].items():
                lines.append(f"| `{name}` | {count} |")
        else:
            lines.append("No missing cells.")

        lines.append("")
        lines.append(
            f"Usable target periods: {summary['first_target_period']} to "
            f"{summary['last_target_period']}"
        )
        lines.append("")
        lines.append(
            f"Windows dropped for a NaN input: {summary['dropped_nan_input']}. "
            f"For a target with no observed district: "
            f"{summary['dropped_unobserved_target']}. "
            f"Kept with a partly masked target: "
            f"{summary['partial_target_windows']}."
        )
        lines.append("")
        lines.append(
            "Dropped target periods: "
            + format_period_ranges(summary["dropped_target_periods"])
        )
        lines.append("")

    lines += [
        "## Output files",
        "",
        "- `data/processed/model_tensors_v0.npz`",
        "- `data/processed/model_tensors_v1.npz`",
        "- `data/processed/model_tensors_v2.npz`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(variant: str, summary: dict[str, object]) -> None:
    """Print one variant's counts to the console."""

    shape = "x".join(str(dim) for dim in summary["shape"])

    print(f"\n{variant}")
    print(f"  X                {shape}")
    print(f"  NaN cells        {summary['nan_cells']}")
    print(
        f"  Windows L=12 h=1 {summary['windows_usable']} usable of "
        f"{summary['windows_total']}"
    )
    print(
        f"  Target periods   {summary['first_target_period']} to "
        f"{summary['last_target_period']}"
    )
    print(
        f"  Dropped targets  "
        + format_period_ranges(summary["dropped_target_periods"])
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Build and write the stage 3 model tensors."""

    try:
        panel = load_panel()
        nodes = load_nodes()
    except FileNotFoundError as error:
        print(error)
        return 1

    check_registry(nodes)
    check_grid(panel, nodes)

    panel = mask_unobserved_weather(panel)
    panel = add_base_features(panel, nodes)
    panel, rolling_names = add_rolling_features(panel)
    panel, history_names = add_history_features(panel)
    adjacency_binary = np.load(PROCESSED_DIR / "adjacency.npz", allow_pickle=True)["A_binary"]
    panel, neighbor_names = add_neighbor_features(panel, adjacency_binary)

    print(f"Panel rows:       {len(panel)}")
    print(f"Reporting periods:{panel['period_id'].nunique():>6}")
    print(f"Districts:        {panel['node_id'].nunique():>6}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, object]] = {}
    built: dict[str, dict[str, np.ndarray]] = {}

    for variant in ("v0", "v1", "v2", "v3"):
        features = feature_names_for(variant, rolling_names, history_names)
        tensors = build_tensors(panel, features)

        output_path = PROCESSED_DIR / f"model_tensors_{variant}.npz"
        np.savez_compressed(
            output_path,
            variant=np.array(variant),
            version=np.array(TENSOR_VERSION),
            lookback_hint=np.array(12),
            **tensors,
        )

        summary = summarise_variant(tensors)
        summaries[variant] = summary
        built[variant] = tensors

        print_summary(variant, summary)
        print(f"  Wrote {output_path.relative_to(PROJECT_DIR)}")

    write_report(summaries, built["v2"], history_names)
    print(f"\nWrote {REPORT_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
