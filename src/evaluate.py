"""Probability-based cross-validation, threshold choice, and model comparison."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, fbeta_score, get_scorer, make_scorer,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import cross_validate


METRICS = (
    "accuracy", "balanced_accuracy", "precision", "recall", "f1",
    "fbeta_2", "roc_auc", "average_precision",
)
THRESHOLD_METRICS = set(METRICS) - {"roc_auc", "average_precision"}


def build_scorers(metrics):
    """Build sklearn scorers, including the course's recall-weighted F2."""
    scorers = {}
    for name in metrics:
        if name not in METRICS:
            raise ValueError(f"Unknown metric: {name}. Choose from {METRICS}")
        if name == "fbeta_2":
            scorers[name] = make_scorer(fbeta_score, beta=2, zero_division=0)
        elif name in {"precision", "recall", "f1"}:
            metric = {"precision": precision_score, "recall": recall_score, "f1": f1_score}[name]
            scorers[name] = make_scorer(metric, zero_division=0)
        else:
            scorers[name] = get_scorer(name)
    return scorers


def cross_validate_pipeline(pipeline, X, y, cv, scorers, selection_metric, n_jobs=1):
    """Score every fold and collect predictions made on unseen validation rows."""
    scores = cross_validate(
        pipeline, X, y, cv=cv, scoring=scorers, return_train_score=True,
        return_estimator=True, return_indices=True, n_jobs=n_jobs,
        error_score="raise",
    )
    folds = pd.DataFrame({"fold": range(1, len(scores[f"test_{selection_metric}"]) + 1)})
    folds["train"] = scores[f"train_{selection_metric}"]
    folds["validation"] = scores[f"test_{selection_metric}"]
    folds["gap"] = folds["train"] - folds["validation"]
    for name in scorers:
        folds[name] = scores[f"test_{name}"]

    oof_proba = np.full(len(X), np.nan)
    seen = np.zeros(len(X), dtype=int)
    for estimator, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        oof_proba[val_idx] = estimator.predict_proba(X.iloc[val_idx])[:, 1]
        seen[val_idx] += 1
    if not np.all(seen == 1):
        raise ValueError("Cross-validation must validate every development row exactly once")
    return folds, oof_proba


def metrics_from_probabilities(y_true, probabilities, threshold=0.5):
    """Compute classification and ranking metrics from the same probabilities."""
    y_pred = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fbeta_2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
    }


def choose_threshold(y_true, probabilities, config):
    """Choose a decision threshold on development predictions, never test rows."""
    objective = config["objective"]
    if objective not in THRESHOLD_METRICS:
        raise ValueError(f"Threshold objective must be one of {sorted(THRESHOLD_METRICS)}")
    low, high, step = config["min"], config["max"], config["step"]
    if not 0 <= low <= high <= 1 or step <= 0:
        raise ValueError("Threshold range must be within [0, 1] with a positive step")
    candidates = np.round(np.arange(low, high + step / 2, step), 6)
    ranked = []
    for threshold in candidates:
        score = metrics_from_probabilities(y_true, probabilities, threshold)[objective]
        ranked.append((score, -abs(threshold - 0.5), -threshold, threshold))
    best = max(ranked)
    return float(best[-1]), float(best[0])


def confusion_counts(y_true, probabilities, threshold):
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def group_false_positive_rates(y_true, probabilities, extras, sensitive_attr, threshold):
    """Group FPRs for model and COMPAS, using only known COMPAS scores."""
    frame = extras.copy()
    frame[sensitive_attr] = frame[sensitive_attr].astype(object).fillna("unknown")
    frame["truth"] = np.asarray(y_true)
    frame["model_pred"] = (np.asarray(probabilities) >= threshold).astype(int)
    result = {}
    for group, rows in frame.groupby(sensitive_attr):
        negative = rows[rows["truth"] == 0]
        known_compas = negative[negative["score_text"].isin(["Low", "Medium", "High"])]
        result[str(group)] = {
            "n": int(len(rows)),
            "negative_n": int(len(negative)),
            "model_fpr": float(negative["model_pred"].mean()) if len(negative) else None,
            "compas_fpr": float((known_compas["score_text"] != "Low").mean())
            if len(known_compas) else None,
        }
    return result


def corrected_paired_test(scores_a, scores_b, test_train_ratio):
    """Nadeau-Bengio correction for overlapping CV training folds."""
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    if len(diff) < 2:
        raise ValueError("At least two paired folds are required")
    se = np.sqrt((1 / len(diff) + test_train_ratio) * diff.var(ddof=1))
    t = float(diff.mean() / se) if se > 0 else (float("inf") if diff.mean() else 0.0)
    p = float(2 * stats.t.sf(abs(t), df=len(diff) - 1))
    return {"mean_diff": float(diff.mean()), "wins": int((diff > 0).sum()), "t": t, "p_value": p}
