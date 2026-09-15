"""
Build walk-forward folds and the per-fold preprocessing statistics.

A single 70/30 split answers one question about one year. Walk-forward answers
it nine times, and it puts the 2017 epidemic in a test fold rather than in the
training data, which is the only way to find out whether an epidemic of that
size is forecastable from climate and case history alone.

Each fold trains on everything before a calendar year and tests on that year:

    fold 1   train to 2016   test 2017
    fold 2   train to 2017   test 2018
    ...
    fold 9   train to 2024   test 2025

The inner validation split is the last calendar year of each fold's training
data. Without it, early stopping would be chosen on the test fold, and the
reported score would be a tuned score dressed up as a held-out one.

Fold years are taken from the calendar's start_date, never from source_year:
the interval labelled 2026 week 53 starts on 2025-12-20, and a source_year fold
boundary would place it a year away from where it belongs.

The preprocessing statistics are fold-dependent, and that is the point of
computing them here rather than freezing them into the tensors. Both are fitted
on periods strictly before the fold's test year:

    imputation   district x month climatology, the method selected in
                 configs/imputation/panel_v1.toml, with a district mean and
                 then a global mean as fallbacks.

    scaling      per-feature z-score over the fitting periods. Not per node:
                 Colombo really does report more cases than Mannar, and scaling
                 that away would remove what the graph is supposed to learn.

Features are imputed. The target never is. A missing case count is carried in
y_mask and excluded from the loss, because an imputed target is a number the
model is rewarded for reproducing and no one ever observed.

Outputs:
    data/processed/folds.json
    data/processed/fold_preprocessing_v0.npz
    data/processed/fold_preprocessing_v1.npz
    results/models/folds_report.md
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]

CALENDAR_PATH = PROJECT_DIR / "data" / "interim" / "reporting_calendar.csv"
PANEL_PATH = PROJECT_DIR / "data" / "processed" / "panel_weekly.parquet"

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "models"

FOLDS_PATH = PROCESSED_DIR / "folds.json"
REPORT_PATH = RESULTS_DIR / "folds_report.md"

FOLDS_VERSION = "folds-v1"

# First and last calendar year used as a test fold. 2017 is the earliest that
# leaves a decade of training data; 2026 is excluded because it is a partial
# year and its last three periods have no ERA5 coverage.
FIRST_TEST_YEAR = 2017
LAST_TEST_YEAR = 2025

N_MONTHS = 12

# Default window geometry used for the reported fold sizes.
DEFAULT_LOOKBACK = 12
DEFAULT_HORIZON = 1


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_calendar(path: Path = CALENDAR_PATH) -> pd.DataFrame:
    """Load the reporting calendar with a calendar year per period."""

    calendar = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    calendar = calendar[["period_id", "start_date", "end_date"]].copy()

    # The fold year is the year the period starts in, taken from the date and
    # not from the source label.
    calendar["year"] = calendar["start_date"].dt.year
    calendar["month"] = calendar["start_date"].dt.month

    return calendar.sort_values("period_id").reset_index(drop=True)


def load_regime_flags(path: Path = PANEL_PATH) -> pd.DataFrame:
    """Load the periods covered by the COVID and 2017 outbreak windows.

    These are evaluation metadata only. They mark which folds need reporting
    separately; they are never model features, because at a forecast origin
    inside 2017 nobody knows that 2017 will be an epidemic year.
    """

    if not path.exists():
        return pd.DataFrame(columns=["period_id", "is_covid_window", "is_2017_outbreak"])

    panel = pd.read_parquet(path)[
        ["period_id", "is_covid_window", "is_2017_outbreak"]
    ]

    return panel.groupby("period_id", as_index=False).max()


def load_tensors(variant: str) -> dict[str, np.ndarray]:
    """Load one model tensor variant."""

    path = PROCESSED_DIR / f"model_tensors_{variant}.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/12.build_model_tensors.py first."
        )

    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------

def year_bounds(calendar: pd.DataFrame, year: int) -> tuple[int, int]:
    """Return the first and last period_id of one calendar year."""

    rows = calendar[calendar["year"] == year]

    if len(rows) == 0:
        raise ValueError(f"No reporting periods start in {year}.")

    return int(rows["period_id"].min()), int(rows["period_id"].max())


def build_folds(
    calendar: pd.DataFrame,
    regime: pd.DataFrame,
    first_test_year: int = FIRST_TEST_YEAR,
    last_test_year: int = LAST_TEST_YEAR,
) -> list[dict]:
    """Return the walk-forward fold definitions."""

    first_period = int(calendar["period_id"].min())
    folds: list[dict] = []

    for fold_id, test_year in enumerate(
        range(first_test_year, last_test_year + 1), start=1
    ):
        test_start, test_end = year_bounds(calendar, test_year)
        val_start, val_end = year_bounds(calendar, test_year - 1)

        fold = {
            "fold_id": fold_id,
            "test_year": test_year,
            "train_start_period": first_period,
            "train_end_period": val_start - 1,
            "val_start_period": val_start,
            "val_end_period": val_end,
            "test_start_period": test_start,
            "test_end_period": test_end,
            # Everything before the test year. Statistics are fitted on this
            # range, inner validation included: at a real forecast origin that
            # history is available, and the validation year is used only to
            # choose a stopping epoch.
            "fit_end_period": test_start - 1,
        }

        fold["covers_covid"] = bool(
            overlaps(regime, "is_covid_window", test_start, test_end)
        )
        fold["covers_2017_outbreak"] = bool(
            overlaps(regime, "is_2017_outbreak", test_start, test_end)
        )
        fold["headline"] = not fold["covers_covid"]

        folds.append(fold)

    return folds


def overlaps(regime: pd.DataFrame, column: str, start: int, end: int) -> bool:
    """Return whether a regime flag is set anywhere in a period range."""

    if column not in regime.columns or len(regime) == 0:
        return False

    window = regime[
        regime["period_id"].between(start, end) & (regime[column] == 1)
    ]

    return len(window) > 0


def check_folds(folds: list[dict], calendar: pd.DataFrame) -> None:
    """Assert the folds are ordered, non-overlapping in test, and leak nothing."""

    last_period = int(calendar["period_id"].max())

    for fold in folds:
        if not (
            fold["train_start_period"]
            <= fold["train_end_period"]
            < fold["val_start_period"]
            <= fold["val_end_period"]
            < fold["test_start_period"]
            <= fold["test_end_period"]
        ):
            raise ValueError(f"Fold {fold['fold_id']} boundaries are not ordered.")

        if fold["fit_end_period"] != fold["test_start_period"] - 1:
            raise ValueError(
                f"Fold {fold['fold_id']} fits statistics into its test range."
            )

        if fold["test_end_period"] > last_period:
            raise ValueError(
                f"Fold {fold['fold_id']} tests beyond the calendar."
            )

    # Test ranges must partition forward in time, or a period would be scored
    # twice and the walk-forward mean would double-count it.
    for earlier, later in zip(folds, folds[1:]):
        if later["test_start_period"] <= earlier["test_end_period"]:
            raise ValueError(
                f"Folds {earlier['fold_id']} and {later['fold_id']} overlap in test."
            )


# ---------------------------------------------------------------------------
# Per-fold statistics
# ---------------------------------------------------------------------------

def fit_fold_statistics(
    tensors: dict[str, np.ndarray],
    months: np.ndarray,
    fit_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit the imputation climatology and the scaler on one fold's history.

    fit_mask selects the periods a fold is allowed to see. Every number returned
    is computed from those periods alone, which is the property the tests check.
    """

    x = tensors["X"]
    n_nodes, n_features = x.shape[1], x.shape[2]

    history = x[fit_mask]
    history_months = months[fit_mask]

    climatology = np.full((n_nodes, N_MONTHS, n_features), np.nan, dtype=np.float32)

    # An all-NaN district-month is expected, not exceptional: it is what the
    # fallback ladder below exists for. numpy warns and returns NaN, which is
    # the behaviour wanted here, so the warning is silenced rather than avoided.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        for month in range(1, N_MONTHS + 1):
            selected = history[history_months == month]
            if len(selected) == 0:
                continue
            climatology[:, month - 1, :] = np.nanmean(selected, axis=0)

        # Fallback ladder. A district-month cell is empty when a district is
        # unobserved for that month across the entire fitting history; the
        # district mean and then the global mean fill it. Both stay fold-local.
        district_mean = np.nanmean(history, axis=0).astype(np.float32)
        global_mean = np.nanmean(history, axis=(0, 1)).astype(np.float32)

    district_mean = np.where(
        np.isnan(district_mean), global_mean[None, :], district_mean
    )
    climatology = np.where(
        np.isnan(climatology), district_mean[:, None, :], climatology
    )

    if np.isnan(climatology).any():
        raise ValueError("Climatology still holds NaN after both fallbacks.")

    # The scaler is fitted on the imputed history, so that filled cells do not
    # shift the mean relative to what the model will actually be fed.
    imputed = impute(history, climatology, history_months)

    mean = imputed.mean(axis=(0, 1)).astype(np.float32)
    std = imputed.std(axis=(0, 1)).astype(np.float32)

    # A constant column has zero variance. Dividing by it produces inf or nan,
    # and the flag channels are constant whenever a fold's history happens to be
    # fully observed.
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

    return {
        "climatology": climatology,
        "district_mean": district_mean,
        "global_mean": global_mean.astype(np.float32),
        "mean": mean,
        "std": std,
    }


def impute(
    x: np.ndarray,
    climatology: np.ndarray,
    months: np.ndarray,
) -> np.ndarray:
    """Fill NaN feature cells from the district x month climatology."""

    filled = x.copy()
    missing = np.isnan(filled)

    if not missing.any():
        return filled

    # climatology is [node, month, feature]; select the month of each period to
    # get a [period, node, feature] replacement aligned with x.
    replacement = climatology[:, months - 1, :].transpose(1, 0, 2)
    filled[missing] = replacement[missing]

    return filled


def transform(
    tensors: dict[str, np.ndarray],
    months: np.ndarray,
    statistics: dict[str, np.ndarray],
) -> np.ndarray:
    """Impute then scale the whole feature cube with one fold's statistics.

    Applied to every period, not only the fitting range: the test year has to be
    transformed with the statistics that were available before it started, which
    is exactly what a deployed model would use.
    """

    filled = impute(tensors["X"], statistics["climatology"], months)

    return ((filled - statistics["mean"]) / statistics["std"]).astype(np.float32)


def split_of(period_id: np.ndarray, fold: dict) -> np.ndarray:
    """Return the split label of each period under one fold."""

    labels = np.full(len(period_id), "unused", dtype=object)

    labels[period_id <= fold["train_end_period"]] = "train"
    labels[
        (period_id >= fold["val_start_period"]) & (period_id <= fold["val_end_period"])
    ] = "val"
    labels[
        (period_id >= fold["test_start_period"])
        & (period_id <= fold["test_end_period"])
    ] = "test"

    return labels


def assign_windows(target_period_id: np.ndarray, fold: dict) -> dict[str, np.ndarray]:
    """Return boolean masks selecting each split's windows by target period.

    A window belongs to the split its target falls in. Its input history may
    reach back into an earlier split, which is correct: at deployment that
    history is available. What must not cross is the fitted statistics, and
    those come from fit_end_period.
    """

    return {
        "train": target_period_id <= fold["train_end_period"],
        "val": (target_period_id >= fold["val_start_period"])
        & (target_period_id <= fold["val_end_period"]),
        "test": (target_period_id >= fold["test_start_period"])
        & (target_period_id <= fold["test_end_period"]),
    }


# ---------------------------------------------------------------------------
# Fold sizing
# ---------------------------------------------------------------------------

def size_folds(
    tensors: dict[str, np.ndarray],
    folds: list[dict],
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """Count usable windows and observed target cells per fold and split."""

    period_id = tensors["period_id"]
    y_mask = tensors["y_mask"]

    origins = np.arange(lookback - 1, len(period_id) - horizon)
    target_index = origins + horizon
    target_period = period_id[target_index]

    # After imputation there are no NaN inputs left, so the only exclusion is a
    # target period with nothing observed anywhere.
    usable = (y_mask[target_index] == 1).any(axis=1)

    records = []
    for fold in folds:
        masks = assign_windows(target_period, fold)
        record = {"fold_id": fold["fold_id"], "test_year": fold["test_year"]}

        for split, mask in masks.items():
            selected = mask & usable
            record[f"{split}_windows"] = int(selected.sum())
            record[f"{split}_observed_cells"] = int(
                y_mask[target_index[selected]].sum()
            )

        record["covers_covid"] = fold["covers_covid"]
        record["covers_2017_outbreak"] = fold["covers_2017_outbreak"]
        records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(
    folds: list[dict],
    calendar: pd.DataFrame,
    sizes: pd.DataFrame,
    filled: dict[str, int],
) -> None:
    """Write the human-readable fold report."""

    dates = calendar.set_index("period_id")

    lines = [
        "# Walk-forward folds",
        "",
        f"Version: `{FOLDS_VERSION}`",
        "",
        "Each fold trains on everything before a calendar year and tests on that",
        "year. Fold years come from `start_date`, never from `source_year`.",
        "",
        "## Folds",
        "",
        "| Fold | Test year | Train | Val | Test | Note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for fold in folds:
        note = []
        if fold["covers_2017_outbreak"]:
            note.append("epidemic")
        if fold["covers_covid"]:
            note.append("COVID, report separately")

        lines.append(
            f"| {fold['fold_id']} | {fold['test_year']} | "
            f"{fold['train_start_period']}-{fold['train_end_period']} | "
            f"{fold['val_start_period']}-{fold['val_end_period']} | "
            f"{fold['test_start_period']}-{fold['test_end_period']} | "
            f"{', '.join(note) if note else 'headline'} |"
        )

    lines += [
        "",
        "## Test fold dates",
        "",
        "| Fold | Test year | From | To |",
        "| --- | --- | --- | --- |",
    ]

    for fold in folds:
        start = dates.loc[fold["test_start_period"], "start_date"].date()
        end = dates.loc[fold["test_end_period"], "end_date"].date()
        lines.append(
            f"| {fold['fold_id']} | {fold['test_year']} | {start} | {end} |"
        )

    lines += [
        "",
        "## Window counts",
        "",
        f"Lookback {DEFAULT_LOOKBACK}, horizon {DEFAULT_HORIZON}, after imputation.",
        "",
        "| Fold | Test year | Train | Val | Test | Test cells |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in sizes.itertuples():
        lines.append(
            f"| {row.fold_id} | {row.test_year} | {row.train_windows} | "
            f"{row.val_windows} | {row.test_windows} | {row.test_observed_cells} |"
        )

    headline = sizes[[not covid for covid in sizes["covers_covid"]]]

    lines += [
        "",
        f"Headline folds (COVID excluded): "
        f"{', '.join(str(f) for f in headline['fold_id'])}",
        "",
        "COVID folds: "
        + ", ".join(
            str(row.fold_id) for row in sizes.itertuples() if row.covers_covid
        ),
        "",
        "## Imputation",
        "",
        "District x month climatology, fitted on each fold's history only.",
        "Cells filled per fold:",
        "",
        "| Variant | Cells filled (fold 1) | Cells filled (fold 9) |",
        "| --- | --- | --- |",
    ]

    for variant, counts in filled.items():
        lines.append(f"| `{variant}` | {counts['first']} | {counts['last']} |")

    lines += [
        "",
        "Targets are never imputed. A missing case count stays NaN, is flagged",
        "in `y_mask`, and is excluded from the loss and from every metric.",
        "",
        "## Output files",
        "",
        "- `data/processed/folds.json`",
        "- `data/processed/fold_preprocessing_v0.npz`",
        "- `data/processed/fold_preprocessing_v1.npz`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Build the folds and the per-fold preprocessing statistics."""

    try:
        calendar = load_calendar()
        regime = load_regime_flags()
        tensor_sets = {variant: load_tensors(variant) for variant in ("v0", "v1")}
    except FileNotFoundError as error:
        print(error)
        return 1

    folds = build_folds(calendar, regime)
    check_folds(folds, calendar)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    FOLDS_PATH.write_text(
        json.dumps({"version": FOLDS_VERSION, "folds": folds}, indent=2),
        encoding="utf-8",
    )

    months = calendar.sort_values("period_id")["month"].to_numpy()

    print(f"Folds:      {len(folds)}")
    print(f"Test years: {folds[0]['test_year']} to {folds[-1]['test_year']}")

    filled: dict[str, dict[str, int]] = {}

    for variant, tensors in tensor_sets.items():
        period_id = tensors["period_id"]

        if len(months) != len(period_id):
            raise ValueError("Calendar and tensor period axes disagree.")

        stacked = {
            "climatology": [],
            "district_mean": [],
            "global_mean": [],
            "mean": [],
            "std": [],
        }
        fill_counts = []

        for fold in folds:
            fit_mask = period_id <= fold["fit_end_period"]

            statistics = fit_fold_statistics(tensors, months, fit_mask)

            for key, value in statistics.items():
                stacked[key].append(value)

            fill_counts.append(int(np.isnan(tensors["X"][fit_mask]).sum()))

        output_path = PROCESSED_DIR / f"fold_preprocessing_{variant}.npz"
        np.savez_compressed(
            output_path,
            version=np.array(FOLDS_VERSION),
            variant=np.array(variant),
            fold_id=np.array([fold["fold_id"] for fold in folds], dtype=np.int32),
            fit_end_period=np.array(
                [fold["fit_end_period"] for fold in folds], dtype=np.int32
            ),
            feature_names=tensors["feature_names"],
            **{key: np.stack(value) for key, value in stacked.items()},
        )

        filled[variant] = {"first": fill_counts[0], "last": fill_counts[-1]}

        print(f"\n{variant}")
        print(f"  Statistics per fold  climatology 25x12x{tensors['X'].shape[2]}")
        print(f"  NaN cells in fold 1 history  {fill_counts[0]}")
        print(f"  NaN cells in fold 9 history  {fill_counts[-1]}")
        print(f"  Wrote {output_path.relative_to(PROJECT_DIR)}")

    sizes = size_folds(tensor_sets["v1"], folds)

    print("\nWindows per fold (v1, L=12, h=1)")
    for row in sizes.itertuples():
        note = " COVID" if row.covers_covid else ""
        note += " epidemic" if row.covers_2017_outbreak else ""
        print(
            f"  fold {row.fold_id}  test {row.test_year}  "
            f"train {row.train_windows:>4}  val {row.val_windows:>3}  "
            f"test {row.test_windows:>3}{note}"
        )

    write_report(folds, calendar, sizes, filled)

    print(f"\nWrote {FOLDS_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
