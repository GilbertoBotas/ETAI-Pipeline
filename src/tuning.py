"""Hyperparameter search and nested cross-validation on development data."""

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score


def tune_pipeline(pipeline, X, y, cv, scoring, search_space, n_trials, random_state, n_jobs=1):
    """Tune a fresh whole pipeline; return its unfitted best configuration and study."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")

    def objective(trial):
        params = {}
        for name, spec in search_space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unknown search-space type for {name}: {spec['type']}")

        candidate = clone(pipeline).set_params(**params)
        scores = cross_val_score(
            candidate, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, error_score="raise"
        )
        trial.set_user_attr("fold_std", float(scores.std(ddof=1)))
        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(
        seed=random_state, n_startup_trials=min(5, n_trials)
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return clone(pipeline).set_params(**study.best_params), study


def nested_cross_validate(
    pipeline, X, y, outer_cv, inner_cv, scoring, search_space,
    n_trials, random_state, n_jobs=1,
):
    """Estimate the tune-then-fit procedure on outer folds unseen by tuning."""
    scorer = get_scorer(scoring)
    rows = []
    y_oof = np.empty(len(y), dtype=np.asarray(y).dtype)

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        tuned, study = tune_pipeline(
            pipeline, X_train, y_train, inner_cv, scoring, search_space,
            n_trials, random_state, n_jobs,
        )
        tuned.fit(X_train, y_train)
        train_score = scorer(tuned, X_train, y_train)
        validation_score = scorer(tuned, X_val, y_val)
        y_oof[val_idx] = tuned.predict(X_val)
        rows.append({
            "fold": fold,
            "train": train_score,
            "validation": validation_score,
            "gap": train_score - validation_score,
            "inner_best": study.best_value,
            **study.best_params,
        })
        print(f"  outer fold {fold}: validation {validation_score:.3f}", flush=True)

    return pd.DataFrame(rows), y_oof


def tuning_report(study, nested_scores, scoring, top=5):
    """Summarize the final search and its inner-versus-outer score difference."""
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    trials = trials.rename(columns=lambda col: col.replace("params_", "").replace("user_attrs_", ""))
    trials = trials.rename(columns={"number": "trial", "value": f"mean_{scoring}"})
    inner = nested_scores["inner_best"].mean()
    outer = nested_scores["validation"].mean()
    lines = [
        f"Final development-set search: {len(study.trials)} trials",
        f"Best hyperparameters: {study.best_params}",
        f"Best trial's inner-CV score: {study.best_value:.3f} (used to choose settings)",
        f"Nested outer-CV score: {outer:.3f} (used to compare models)",
        f"Mean inner-best minus outer score: {inner - outer:+.3f}",
        "",
        f"Top {top} trials:",
        trials.sort_values(f"mean_{scoring}", ascending=False).head(top).to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        ),
    ]
    return "\n".join(lines)
