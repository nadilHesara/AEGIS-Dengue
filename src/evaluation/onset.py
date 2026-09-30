"""
Outbreak onset as its own prediction task (revision of 29 September 2026).

One definition, used by every comparator:

    at-risk origin   district n at origin week t whose last QUIET_WEEKS weeks
                     (t - 3 .. t) are all observed and below the district's
                     outbreak threshold. Only at-risk origins are scored.
    label (window K) whether the first threshold crossing after t happens in
                     t + 1 .. t + K. Because the origin is quiet, "first
                     crossing" and "any crossing" coincide inside the window.
                     An origin is eligible only if all K target weeks are
                     observed and lie in the split being scored.
    onset event      a week s at or above threshold whose preceding
                     QUIET_WEEKS weeks were observed and quiet. Each event is
                     counted once for event-level scores; it is detected if an
                     alarm fired at an eligible origin t in s - K .. s - 1, and
                     its lead time is s minus the earliest such t.

The threshold is scripts/15's peak threshold (90th percentile of the
district's pre-test history), a statistical definition, not an official
epidemic threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUIET_WEEKS = 4
WINDOWS = (4, 8)
ALARM_BUDGET = 0.10  # alarms on 10% of at-risk origins, cutoff set on the previous year


def quiet_mask(y: np.ndarray, mask: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    return (mask == 1) & (y < threshold[None, :])


def above_mask(y: np.ndarray, mask: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    return (mask == 1) & (y >= threshold[None, :])


def at_risk(y: np.ndarray, mask: np.ndarray, threshold: np.ndarray,
            quiet_weeks: int = QUIET_WEEKS) -> np.ndarray:
    """[periods, nodes] True where t - quiet_weeks + 1 .. t are all observed and quiet."""

    quiet = quiet_mask(y, mask, threshold)
    out = np.zeros_like(quiet)
    for t in range(quiet_weeks - 1, len(y)):
        out[t] = quiet[t - quiet_weeks + 1 : t + 1].all(axis=0)
    return out


def window_label(y: np.ndarray, mask: np.ndarray, threshold: np.ndarray,
                 window: int) -> tuple[np.ndarray, np.ndarray]:
    """(label, complete), both [periods, nodes].

    label[t] = any crossing in t+1..t+window; complete[t] = all of those weeks
    exist and are observed.
    """

    above = above_mask(y, mask, threshold)
    n = len(y)
    label = np.zeros_like(above)
    complete = np.zeros_like(above)
    for t in range(n - window):
        label[t] = above[t + 1 : t + window + 1].any(axis=0)
        complete[t] = (mask[t + 1 : t + window + 1] == 1).all(axis=0)
    return label, complete


def onset_events(y: np.ndarray, mask: np.ndarray, threshold: np.ndarray,
                 quiet_weeks: int = QUIET_WEEKS) -> np.ndarray:
    """[periods, nodes] True at the first above-threshold week after a quiet run."""

    above = above_mask(y, mask, threshold)
    quiet = quiet_mask(y, mask, threshold)
    out = np.zeros_like(above)
    for s in range(quiet_weeks, len(y)):
        out[s] = above[s] & quiet[s - quiet_weeks : s].all(axis=0)
    return out


def eligible_origins(y, mask, threshold, window, period_id, first_target, last_target):
    """[periods, nodes]: at-risk, window complete, and every target in [first, last]."""

    risk = at_risk(y, mask, threshold)
    _, complete = window_label(y, mask, threshold, window)
    n = len(y)
    inside = np.zeros(n, dtype=bool)
    for t in range(n - window):
        inside[t] = (period_id[t + 1] >= first_target) and (period_id[t + window] <= last_target)
    return risk & complete & inside[:, None]


def alarm_cutoff(previous_scores: np.ndarray, budget: float = ALARM_BUDGET) -> float:
    """Score above which `budget` of the previous year's eligible origins alarm."""

    if len(previous_scores) == 0:
        return np.nan
    return float(np.quantile(previous_scores, 1 - budget))


def event_detection(alarm: np.ndarray, eligible: np.ndarray, events: np.ndarray,
                    window: int, event_weeks: np.ndarray) -> pd.DataFrame:
    """One row per onset event in `event_weeks` (bool over periods) that has at
    least one eligible origin before it: detected, and lead time in weeks."""

    rows = []
    n_periods = alarm.shape[0]
    for s, node in zip(*np.nonzero(events & event_weeks[:, None])):
        origins = np.arange(max(s - window, 0), s)
        origins = origins[origins < n_periods]
        candidates = origins[eligible[origins, node]]
        if not len(candidates):
            continue
        fired = candidates[alarm[candidates, node]]
        rows.append({"period_index": int(s), "node_id": int(node),
                     "detected": bool(len(fired)),
                     "lead_weeks": float(s - fired.min()) if len(fired) else np.nan})
    return pd.DataFrame(rows, columns=["period_index", "node_id", "detected", "lead_weeks"])


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    from scipy import stats

    label = label.astype(bool)
    positives, negatives = label.sum(), (~label).sum()
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = stats.rankdata(score)
    return float((ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def pr_auc(score: np.ndarray, label: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(label, score)) if label.any() else np.nan
