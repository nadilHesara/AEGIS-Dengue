"""
Measure the climate-to-dengue delay directly, before any model learns one.

The baseline asserts a delay: trailing 4, 8 and 12 period means, chosen by hand
and identical for all 25 districts. The lag encoder in scripts/18 learns one.
Neither of those is a measurement, and without a measurement there is no way to
tell a learned kernel that found something from a learned kernel that drifted
somewhere and stopped.

So this script computes the thing itself. For every district and every climate
feature it correlates the feature at t - lag against log1p cases at t, for lags
0 to 25, and reports where the association is strongest. It trains nothing.

Two versions of every correlation are reported, and the gap between them is the
point:

    raw             the series as they are

    deseasonalised  both series with their district x week-of-year mean
                    removed, so what is left is "this week was wetter than a
                    normal week w" against "cases were higher than a normal
                    week w"

Dengue and rainfall are both strongly seasonal in Sri Lanka, and there are two
monsoons roughly half a year apart. A raw correlation therefore peaks wherever
the two seasonal cycles happen to line up, and will do so at several lags at
once -- including lags that are half a cycle away and have no causal reading.
The deseasonalised correlation is the one that answers "does extra rain lead to
extra dengue, and how long does it take". Where the two disagree, the raw peak
is a calendar artefact.

Every statistic is fitted on a fold's own history: `period_id <= fit_end_period`.
No test year touches this, so the peaks can be used to initialise the encoder,
and later compared against what the encoder learned, without leaking.

Outputs:
    results/eda/lag_correlation.csv
    results/eda/lag_correlation_report.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "eda"

FOLDS_PATH = PROCESSED_DIR / "folds.json"
NODES_PATH = PROCESSED_DIR / "nodes.csv"

REPORT_PATH = RESULTS_DIR / "lag_correlation_report.md"
SCAN_PATH = RESULTS_DIR / "lag_correlation.csv"

SCAN_VERSION = "lag-scan-v1"

# Lags 0..25, matching the encoder's 26-period reach so the two are directly
# comparable. Raise it with --max-lag to ask whether the true delay runs past
# the reach, which is a different question and worth asking separately.
DEFAULT_MAX_LAG = 25

# The climate channels the encoder smooths. Case history, seasonality, the
# observation flags and the centroids are not delayed quantities.
CLIMATE_FEATURES = (
    "rainfall_daily_mean_mm",
    "rainy_days_frac",
    "temperature_mean_c",
    "diurnal_range_c",
    "dewpoint_mean_c",
    "relative_humidity_mean",
    "wind_speed_mean",
)

# A correlation on fewer pairs than this is not reported.
MIN_PAIRS = 30

# Below this, a peak lag is the argmax of noise rather than a delay.
#
# The naive iid band on ~900 weekly pairs is 1.96/sqrt(900) = 0.065, but weekly
# case and climate series are heavily autocorrelated, so the effective sample
# size is a small fraction of the nominal one and the true band is several
# times wider. 0.10 is a deliberately blunt floor, not a significance test; the
# scan reports the correlation so the reader can apply their own.
WEAK_CORRELATION = 0.10


def load_module(name: str, filename: str):
    """Load one of the numbered pipeline scripts as a module."""

    spec = importlib.util.spec_from_file_location(
        name, PROJECT_DIR / "scripts" / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


folds_module = load_module("build_folds", "features/14.build_folds.py")


# ---------------------------------------------------------------------------
# Series preparation
# ---------------------------------------------------------------------------

def week_of_year(start_date: np.ndarray) -> np.ndarray:
    """Return the ISO week of each reporting period. [T]

    Week rather than the month used by the fold climatology: a month bucket is
    four to five periods wide, which smears exactly the resolution a lag scan
    exists to measure.
    """

    return (
        pd.DatetimeIndex(start_date).isocalendar().week.to_numpy().astype(np.int64)
    )


def deseasonalise(
    series: np.ndarray, mask: np.ndarray, weeks: np.ndarray
) -> np.ndarray:
    """Subtract the district x week-of-year mean. [T, N] -> [T, N]

    Fitted on the observed cells of whatever periods are passed in, which the
    caller has already restricted to a fold's history. A week with no observed
    value in a district keeps its raw value; with 17 years of history that is
    rare, and forcing it to zero would invent an anomaly.
    """

    out = series.copy()

    for week in np.unique(weeks):
        rows = weeks == week
        selected = np.where(mask[rows], series[rows], np.nan)

        with np.errstate(invalid="ignore"):
            mean = np.nanmean(selected, axis=0)

        mean = np.where(np.isnan(mean), 0.0, mean)
        out[rows] = series[rows] - mean[None, :]

    return out


def masked_correlation(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlation down axis 0, per column, over masked cells.

    Returns (correlation [N], pair count [N]). A column with fewer than
    MIN_PAIRS usable pairs, or with no variance in either series, returns NaN
    rather than a number that would be read as a measurement.
    """

    weight = mask.astype(np.float64)
    count = weight.sum(axis=0)

    x = np.where(mask, x, 0.0).astype(np.float64)
    y = np.where(mask, y, 0.0).astype(np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_x = x.sum(axis=0) / count
        mean_y = y.sum(axis=0) / count

        dx = np.where(mask, x - mean_x[None, :], 0.0)
        dy = np.where(mask, y - mean_y[None, :], 0.0)

        covariance = (dx * dy).sum(axis=0)
        spread = np.sqrt((dx**2).sum(axis=0) * (dy**2).sum(axis=0))

        correlation = np.where(spread > 0, covariance / spread, np.nan)

    return np.where(count >= MIN_PAIRS, correlation, np.nan), count


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------

def scan_fold(
    tensors: dict[str, np.ndarray],
    weeks: np.ndarray,
    fold: dict,
    feature_indices: dict[str, int],
    max_lag: int,
) -> pd.DataFrame:
    """Correlate every district x feature x lag on one fold's history."""

    history = tensors["period_id"] <= fold["fit_end_period"]

    cases = np.log1p(tensors["y"][history])
    case_mask = tensors["y_mask"][history] == 1
    # A masked target is still NaN in `y`; it is excluded everywhere below, but
    # it has to be finite first or it would poison the sums.
    cases = np.nan_to_num(cases, nan=0.0)

    fold_weeks = weeks[history]
    cases_anomaly = deseasonalise(cases, case_mask, fold_weeks)

    rows = []
    for feature, index in feature_indices.items():
        climate = tensors["X"][history, :, index]
        climate_mask = np.isfinite(climate)
        climate = np.nan_to_num(climate, nan=0.0)

        climate_anomaly = deseasonalise(climate, climate_mask, fold_weeks)

        for lag in range(max_lag + 1):
            # climate at t - lag against cases at t. Slicing rather than
            # rolling: no wrap-around, and the alignment is visible.
            past = slice(0, len(cases) - lag)
            now = slice(lag, len(cases))

            joint = climate_mask[past] & case_mask[now]

            raw, pairs = masked_correlation(
                climate[past], cases[now], joint
            )
            anomaly, _ = masked_correlation(
                climate_anomaly[past], cases_anomaly[now], joint
            )

            rows.append(
                pd.DataFrame(
                    {
                        "fold_id": fold["fold_id"],
                        "test_year": fold["test_year"],
                        "fit_end_period": fold["fit_end_period"],
                        "node_id": tensors["node_id"],
                        "canonical_name": tensors["canonical_name"],
                        "feature": feature,
                        "lag": lag,
                        "correlation": raw,
                        "correlation_deseasonalised": anomaly,
                        "pairs": pairs.astype(np.int32),
                    }
                )
            )

    return pd.concat(rows, ignore_index=True)


def add_peaks(scan: pd.DataFrame) -> pd.DataFrame:
    """Attach the peak and trough lag of each district x feature curve.

    Both, and signed, rather than one argmax over the absolute correlation.
    These curves are not single-humped: a driver that is positively associated
    at its causal delay is negatively associated roughly half a seasonal cycle
    away, purely because the two calendars fall out of phase. In Colombo the
    rainfall trough at lag 23 is very slightly deeper than the peak at lag 5,
    so an absolute argmax reports 23 -- an anti-correlation with no causal
    reading -- and buries the number the scan exists to produce.

    `peak_lag` is therefore where the association is most positive, which is
    the hypothesis for rainfall, temperature and humidity. `trough_lag` is
    where it is most negative, which is where a feature like wind speed should
    be read. Carrying both means neither has to be guessed at.
    """

    scan = scan.copy()
    keys = ["fold_id", "node_id", "feature"]
    grouping = [scan[key] for key in keys]

    for column, suffix in (
        ("correlation", ""),
        ("correlation_deseasonalised", "_deseasonalised"),
    ):
        for extreme, label in (("idxmax", "peak"), ("idxmin", "trough")):
            index = getattr(scan[column].groupby(grouping), extreme)()

            found = scan.loc[index.dropna().astype(int), keys + ["lag", column]]
            found = found.rename(
                columns={
                    "lag": f"{label}_lag{suffix}",
                    column: f"{label}_correlation{suffix}",
                }
            )

            scan = scan.merge(found, on=keys, how="left")

    return scan


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_alignment(scan: pd.DataFrame) -> list[str]:
    """Flag the failure modes that mean the scan is not measuring what it says.

    These are diagnostics, not assertions. A scan that trips one is still worth
    reading -- the point is that it is read as a warning and not as a delay.
    """

    warnings_found = []

    rainfall = scan[
        (scan["feature"] == "rainfall_daily_mean_mm") & (scan["lag"] == 0)
    ]
    at_zero = (rainfall["peak_lag_deseasonalised"] == 0).mean()
    if at_zero > 0.5:
        warnings_found.append(
            f"Deseasonalised rainfall peaks at lag 0 in {100 * at_zero:.0f}% of "
            "district-folds. Rain cannot cause a reported case in the same "
            "week; check the panel alignment in scripts/10 before using this."
        )

    max_lag = scan["lag"].max()
    at_edge = (rainfall["peak_lag_deseasonalised"] >= max_lag - 1).mean()
    if at_edge > 0.5:
        warnings_found.append(
            f"Deseasonalised rainfall peaks at lag {max_lag - 1} or beyond in "
            f"{100 * at_edge:.0f}% of district-folds. The true delay may run "
            f"past the scanned reach; re-run with --max-lag above {max_lag}."
        )

    thin = (scan["pairs"] < MIN_PAIRS).mean()
    if thin > 0.01:
        warnings_found.append(
            f"{100 * thin:.1f}% of cells had fewer than {MIN_PAIRS} usable "
            "pairs and were dropped."
        )

    # A peak lag is only a measurement if there is a peak. Where the whole
    # curve sits inside the noise, the argmax is picking the largest of 26
    # coin flips and will move from fold to fold for that reason alone.
    weak = (
        scan.drop_duplicates(["fold_id", "node_id", "feature"])
        .groupby("feature")["peak_correlation_deseasonalised"]
        .median()
    )

    for feature, strength in weak.items():
        if abs(strength) < WEAK_CORRELATION:
            warnings_found.append(
                f"`{feature}`: median deseasonalised peak correlation is only "
                f"{strength:+.3f}. Below {WEAK_CORRELATION:.2f} the peak lag is "
                "not resolvable and should not be quoted as a delay."
            )

    return warnings_found


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def feature_summary(scan: pd.DataFrame, fold_id: int) -> pd.DataFrame:
    """Spread of the peak lag across districts, one row per feature."""

    peaks = scan[scan["fold_id"] == fold_id].drop_duplicates(
        ["node_id", "feature"]
    )

    return (
        peaks.groupby("feature")
        .agg(
            raw_median=("peak_lag", "median"),
            raw_min=("peak_lag", "min"),
            raw_max=("peak_lag", "max"),
            deseasonalised_median=("peak_lag_deseasonalised", "median"),
            deseasonalised_min=("peak_lag_deseasonalised", "min"),
            deseasonalised_max=("peak_lag_deseasonalised", "max"),
            strength=("peak_correlation_deseasonalised", "mean"),
            trough_median=("trough_lag_deseasonalised", "median"),
            trough_strength=("trough_correlation_deseasonalised", "mean"),
        )
        .reset_index()
        .sort_values("deseasonalised_median")
    )


def stability(scan: pd.DataFrame) -> pd.DataFrame:
    """How much each feature's peak lag moves from fold to fold."""

    peaks = scan.drop_duplicates(["fold_id", "node_id", "feature"])

    spread = (
        peaks.groupby(["node_id", "feature"])["peak_lag_deseasonalised"]
        .std()
        .reset_index()
    )

    return (
        spread.groupby("feature")["peak_lag_deseasonalised"]
        .agg(["mean", "max"])
        .reset_index()
        .rename(columns={"mean": "mean_sd", "max": "worst_sd"})
        .sort_values("mean_sd")
    )


def write_report(scan: pd.DataFrame, fold_id: int, alerts: list[str]) -> None:
    """Write the scan report."""

    max_lag = int(scan["lag"].max())
    summary = feature_summary(scan, fold_id)

    lines = [
        "# Measured climate lags",
        "",
        "Cross-correlation of each climate feature at `t - lag` against",
        "`log1p(cases)` at `t`, per district. Nothing is trained here; this is",
        "the yardstick the hand-coded windows and the learned kernels are both",
        "checked against.",
        "",
        f"Version: `{SCAN_VERSION}`",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Lags scanned | 0 to {max_lag} |",
        f"| Folds | {sorted(scan['fold_id'].unique().tolist())} |",
        f"| Districts | {scan['node_id'].nunique()} |",
        f"| Features | {scan['feature'].nunique()} |",
        f"| Minimum pairs | {MIN_PAIRS} |",
        "",
        "Every correlation uses `period_id <= fit_end_period` for its fold, so",
        "no test year contributes to any number below.",
        "",
    ]

    if alerts:
        lines += ["## Warnings", ""]
        lines += [f"- {alert}" for alert in alerts]
        lines += [""]

    lines += [
        "## Peak lag by feature",
        "",
        f"Fold {fold_id} (the longest history). `raw` is the correlation as-is;",
        "`deseasonalised` removes the district x week-of-year mean from both",
        "series first. Where the two disagree, the raw peak is the seasonal",
        "cycles lining up rather than a delay.",
        "",
        "`peak` is the most positive association, which is the hypothesis for",
        "rainfall, temperature and humidity; `trough` is the most negative, and",
        "is where a feature like wind speed should be read.",
        "",
        "| Feature | Raw peak | Deseas. peak | Deseas. range | Mean r | Trough | Mean r |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in summary.itertuples():
        lines.append(
            f"| `{row.feature}` | {row.raw_median:.0f} | "
            f"{row.deseasonalised_median:.0f} | "
            f"{row.deseasonalised_min:.0f}-{row.deseasonalised_max:.0f} | "
            f"{row.strength:+.3f} | {row.trough_median:.0f} | "
            f"{row.trough_strength:+.3f} |"
        )

    rainfall = (
        scan[
            (scan["fold_id"] == fold_id)
            & (scan["feature"] == "rainfall_daily_mean_mm")
        ]
        .drop_duplicates(["node_id"])
        .sort_values("peak_lag_deseasonalised")
    )

    lines += [
        "",
        "## Rainfall lag by district",
        "",
        f"Fold {fold_id}, deseasonalised. This is the map the learned kernels",
        "have to reproduce to be believable.",
        "",
        "| District | Peak lag | r at peak | Raw peak lag |",
        "| --- | --- | --- | --- |",
    ]

    for row in rainfall.itertuples():
        lines.append(
            f"| {row.canonical_name} | {row.peak_lag_deseasonalised:.0f} | "
            f"{row.peak_correlation_deseasonalised:+.3f} | {row.peak_lag:.0f} |"
        )

    if scan["fold_id"].nunique() > 1:
        spread = stability(scan)

        lines += [
            "",
            "## Stability across folds",
            "",
            "Standard deviation of a district's deseasonalised peak lag across",
            "folds, in periods. A feature whose peak moves several periods",
            "between folds is not a delay anyone should quote.",
            "",
            "| Feature | Mean sd | Worst district sd |",
            "| --- | --- | --- |",
        ]

        for row in spread.itertuples():
            lines.append(
                f"| `{row.feature}` | {row.mean_sd:.1f} | {row.worst_sd:.1f} |"
            )

    lines += [
        "",
        "## Output files",
        "",
        f"- `{SCAN_PATH.relative_to(PROJECT_DIR).as_posix()}`",
        f"- `{REPORT_PATH.relative_to(PROJECT_DIR).as_posix()}`",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    arguments = parser.parse_args()

    tensors = folds_module.load_tensors("v0")

    nodes = pd.read_csv(NODES_PATH).sort_values("node_id")
    tensors["canonical_name"] = nodes["canonical_name"].to_numpy(dtype=object)

    feature_names = list(tensors["feature_names"])
    missing = [name for name in CLIMATE_FEATURES if name not in feature_names]
    if missing:
        raise ValueError(f"Missing climate features in v0 tensors: {missing}.")

    feature_indices = {name: feature_names.index(name) for name in CLIMATE_FEATURES}

    weeks = week_of_year(tensors["start_date"])

    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["folds"]
    if arguments.folds:
        folds = [fold for fold in folds if fold["fold_id"] in arguments.folds]

    print(f"Folds:     {[fold['fold_id'] for fold in folds]}")
    print(f"Districts: {len(tensors['node_id'])}")
    print(f"Features:  {len(CLIMATE_FEATURES)}")
    print(f"Lags:      0 to {arguments.max_lag}\n")

    scan = pd.concat(
        [
            scan_fold(tensors, weeks, fold, feature_indices, arguments.max_lag)
            for fold in folds
        ],
        ignore_index=True,
    )
    scan = add_peaks(scan)

    alerts = check_alignment(scan)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scan.to_csv(SCAN_PATH, index=False)

    report_fold = max(fold["fold_id"] for fold in folds)
    write_report(scan, report_fold, alerts)

    summary = feature_summary(scan, report_fold)

    print(f"{'feature':<24} {'raw':>6} {'deseas':>8} {'range':>10} {'mean r':>8}")
    for row in summary.itertuples():
        span = (
            f"{row.deseasonalised_min:.0f}-{row.deseasonalised_max:.0f}"
        )
        print(
            f"{row.feature:<24} {row.raw_median:>6.0f} "
            f"{row.deseasonalised_median:>8.0f} {span:>10} {row.strength:>+8.3f}"
        )

    for alert in alerts:
        print(f"\nWARNING: {alert}")

    print(f"\nWrote {SCAN_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
