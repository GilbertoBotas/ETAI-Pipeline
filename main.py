"""Compare all configured models and append their results to one CSV.

Run with ``python main.py`` from the repository root.
"""

import hashlib
import json
import atexit
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import optuna
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.evaluate import (
    build_scorers, choose_threshold, confusion_counts, corrected_paired_test,
    cross_validate_pipeline, group_false_positive_rates, metrics_from_probabilities,
)
from src.model import build_pipeline
from src.preprocessing import add_engineered_features, clean_dataset, split_dev_test, split_features_target
from src.results import save_run_report, save_scores
from src.tuning import nested_cross_validate, tune_pipeline


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def experiment_id(config):
    """Identify the data, features, cleaning, and folds behind paired scores."""
    data_hash = hashlib.sha256(Path(config["data"]["path"]).read_bytes()).hexdigest()
    protocol = {
        key: config[key] for key in
        ("data", "diagnostics", "preprocessing", "feature_engineering", "test_set", "cv", "threshold")
    }
    payload = json.dumps({"data_hash": data_hash, "protocol": protocol}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main():
    config = load_config()
    selection_config = config.get("feature_selection", {})
    cache_dir = tempfile.mkdtemp(prefix="pipeline_cache_")
    atexit.register(shutil.rmtree, cache_dir, ignore_errors=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    raw = load_data(config["data"]["path"])
    clean = clean_dataset(raw, config["diagnostics"])
    feature_config = config.get("feature_engineering", {})
    if feature_config.get("enabled", False):
        clean = add_engineered_features(clean)
    else:
        engineered = set(feature_config.get("columns", []))
        config["preprocessing"]["numeric_features"] = [
            column for column in config["preprocessing"]["numeric_features"]
            if column not in engineered
        ]
    X, y, extras = split_features_target(
        clean, config["data"], config["preprocessing"].get("mnar_indicator_sources", [])
    )
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras, config["test_set"]["size"], config["test_set"]["random_state"]
    )

    cv_config = config["cv"]
    metric_names = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    if selection_metric not in metric_names:
        raise ValueError("cv.selection_metric must also appear in cv.metrics")
    scorers = build_scorers(metric_names)
    outer_cv = StratifiedKFold(
        n_splits=cv_config["n_splits"], shuffle=cv_config["shuffle"],
        random_state=cv_config["random_state"] if cv_config["shuffle"] else None,
    )
    tuning = config.get("tuning", {})
    if tuning.get("enabled", False):
        inner_cv = StratifiedKFold(
            n_splits=tuning["n_splits"], shuffle=True, random_state=tuning["random_state"]
        )

    print(f"Development rows: {len(X_dev)} | test rows reserved: {len(X_test)}")
    print(f"Selection metric: {selection_metric} | {outer_cv.n_splits} outer folds", flush=True)
    evaluated = []
    for candidate in config["models"]:
        name = candidate["type"]
        candidate_selection = {
            **selection_config,
            "enabled": selection_config.get("enabled", False)
            and name in selection_config.get("models", [name]),
        }
        pipeline = build_pipeline(
            config["preprocessing"], candidate, candidate_selection, memory=cache_dir
        )
        defaults, default_proba = cross_validate_pipeline(
            pipeline, X_dev, y_dev, outer_cv, scorers, selection_metric, cv_config["n_jobs"]
        )
        folds, oof_proba = defaults, default_proba
        fitted_recipe = pipeline
        best_params = {}
        best_trial = None
        trials = 0

        if tuning.get("enabled", False) and name in tuning["search_spaces"]:
            trials = tuning["n_trials"][name]
            search_space = dict(tuning["search_spaces"][name])
            if candidate_selection["enabled"]:
                search_space.update(candidate_selection.get("search_space", {}))
            print(f"Tuning {name}: {trials} trials, {tuning['n_splits']} inner folds", flush=True)
            folds, oof_proba = nested_cross_validate(
                pipeline, X_dev, y_dev, outer_cv, inner_cv, scorers, selection_metric,
                search_space, trials, tuning["random_state"],
                cv_config["n_jobs"],
            )
            fitted_recipe, study = tune_pipeline(
                pipeline, X_dev, y_dev, inner_cv, scorers[selection_metric],
                search_space, trials, tuning["random_state"],
                cv_config["n_jobs"],
            )
            best_params, best_trial = study.best_params, study.best_value

        evaluated.append({
            "candidate": candidate,
            "selection": candidate_selection,
            "folds": folds,
            "oof_proba": oof_proba,
            "recipe": fitted_recipe,
            "default_mean": float(defaults["validation"].mean()),
            "best_params": best_params,
            "best_trial": best_trial,
            "trials": trials,
        })
        print(f"  {name}: {selection_metric} {folds['validation'].mean():.3f} "
              f"+/- {folds['validation'].std(ddof=1):.3f}", flush=True)

    winner = max(evaluated, key=lambda item: item["folds"]["validation"].mean())
    winner_name = winner["candidate"]["type"]
    floor = next(item for item in evaluated if item["candidate"]["type"] == "dummy")
    floor_mean = float(floor["folds"]["validation"].mean())
    threshold, threshold_score = choose_threshold(
        y_dev, winner["oof_proba"], config["threshold"]
    )
    print(f"Selected: {winner_name} | threshold: {threshold:.2f} "
          f"({config['threshold']['objective']}={threshold_score:.3f} on development OOF)")

    # Refit only the selected model; the test rows never choose a candidate or threshold.
    final_model = winner["recipe"].fit(X_dev, y_dev)
    kept_features = (
        final_model.named_steps["prep"].get_feature_names_out()[
            final_model.named_steps["select"].get_support()
        ].tolist() if "select" in final_model.named_steps else
        final_model.named_steps["prep"].get_feature_names_out().tolist()
    )
    print("Features kept: " + ", ".join(kept_features))
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_metrics = metrics_from_probabilities(y_test, test_proba, threshold)
    print("Final test metrics: " + ", ".join(
        f"{key}={test_metrics[key]:.3f}" for key in metric_names
    ))

    protocol_id = experiment_id(config)
    rows = []
    for item in evaluated:
        candidate = item["candidate"]
        item_selection = item["selection"]
        name = candidate["type"]
        folds = item["folds"]
        selected = name == winner_name
        paired = corrected_paired_test(
            winner["folds"]["validation"], folds["validation"],
            1 / (outer_cv.n_splits - 1),
        )
        # Threshold-dependent OOF metrics here use 0.5 for comparable candidate rows.
        oof_metrics = metrics_from_probabilities(y_dev, item["oof_proba"], 0.5)
        row = {
            "run_id": run_id,
            "experiment_id": protocol_id,
            "data": config["data"]["path"],
            "model": name,
            "selected": selected,
            "tuned": bool(item["trials"]),
            "class_weight": candidate.get("params", {}).get("class_weight", ""),
            "encoder": config["preprocessing"]["encoder"],
            "scaler": config["preprocessing"]["scaler"],
            "feature_selection": json.dumps({
                "enabled": bool(item_selection["enabled"]),
                "min_votes": item["best_params"].get(
                    "select__min_votes", item_selection.get("min_votes")
                ) if item_selection["enabled"] else None,
            }, sort_keys=True),
            "base_params": json.dumps(candidate.get("params", {}), sort_keys=True),
            "tuned_params": json.dumps(item["best_params"], sort_keys=True),
            "trials": item["trials"],
            "inner_folds": tuning.get("n_splits", 0) if item["trials"] else 0,
            "outer_folds": outer_cv.n_splits,
            "metric": selection_metric,
            "mean": float(folds["validation"].mean()),
            "std": float(folds["validation"].std(ddof=1)),
            "train_mean": float(folds["train"].mean()),
            "gap_mean": float(folds["gap"].mean()),
            "untuned": item["default_mean"],
            "floor": floor_mean,
            "best_trial_inner_score": item["best_trial"],
            "paired_mean_advantage_of_selected": paired["mean_diff"],
            "paired_wins_of_selected": paired["wins"],
            "paired_p_value": paired["p_value"],
            "decision_threshold": threshold if selected else 0.5,
            "threshold_objective": config["threshold"]["objective"] if selected else "default",
            "threshold_objective_oof_score": threshold_score if selected else None,
            **{f"fold_{index}": float(score) for index, score in enumerate(folds["validation"], 1)},
            **{f"cv_{metric}_mean": float(folds[metric].mean()) for metric in metric_names},
            **{f"cv_{metric}_std": float(folds[metric].std(ddof=1)) for metric in metric_names},
            **{f"oof_{metric}_at_0_5": float(value) for metric, value in oof_metrics.items()},
            **{f"oof_{key}_at_0_5": value for key, value in confusion_counts(
                y_dev, item["oof_proba"], 0.5
            ).items()},
            "oof_group_fpr_at_0_5": json.dumps(group_false_positive_rates(
                y_dev, item["oof_proba"], extras_dev, config["data"]["sensitive_attr"], 0.5
            ), sort_keys=True),
        }
        if selected:
            row["features_kept"] = json.dumps(kept_features)
            row.update({f"test_{metric}": float(value) for metric, value in test_metrics.items()})
            row.update({f"test_{key}": value for key, value in confusion_counts(
                y_test, test_proba, threshold
            ).items()})
            row["test_group_fpr"] = json.dumps(group_false_positive_rates(
                y_test, test_proba, extras_test, config["data"]["sensitive_attr"], threshold
            ), sort_keys=True)
        rows.append(row)

    # The corrected t-test handles shared CV training rows; Holm also accounts
    # for comparing the selected model against several alternatives.
    competitors = sorted(
        (index for index, row in enumerate(rows) if not row["selected"]),
        key=lambda index: rows[index]["paired_p_value"],
    )
    running_adjusted = 0.0
    for rank, index in enumerate(competitors):
        adjusted = min(1.0, (len(competitors) - rank) * rows[index]["paired_p_value"])
        running_adjusted = max(running_adjusted, adjusted)
        rows[index]["paired_p_holm"] = running_adjusted
    next(row for row in rows if row["selected"])["paired_p_holm"] = 1.0

    destination = config["output"]["scoreboard"]
    board = save_scores(destination, rows)
    current = board[board["run_id"] == run_id].sort_values("mean", ascending=False)
    print("\nThis run (paired comparisons use these same outer folds):")
    print(current[["model", "tuned", "mean", "std", "untuned", "floor",
                   "paired_p_value", "paired_p_holm"]].to_string(
                       index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"All model results saved to {destination}")
    save_run_report(
        destination, run_id, winner_name,
        next(row for row in rows if row["selected"])["feature_selection"],
        kept_features, selection_metric, winner["folds"]["validation"].mean(),
    )
    shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
