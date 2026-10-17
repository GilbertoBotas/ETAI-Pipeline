"""
Entry point for cross-validation, tuning, and final evaluation.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load -> clean -> split -> compare defaults -> nested CV of tuned models
    -> tune/refit the winner on development rows -> held-out evaluation
"""
import optuna
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import clean_dataset, split_features_target, split_dev_test
from src.model import build_pipeline
from src.evaluate import cross_validate_pipeline, comparison_report, evaluate, fairness_report
from src.results import save_run
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report



def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    # Hold out the test rows before fitting or comparing any candidate models.
    X_dev, X_test, y_dev, y_test, _, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    cv_config = config["cv"]
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config["shuffle"],
        random_state=cv_config["random_state"] if cv_config["shuffle"] else None,
    )

    rows = []
    fold_reports = []
    for candidate in config["models"]:
        folds = cross_validate_pipeline(
            build_pipeline(config["preprocessing"], candidate), X_dev, y_dev, cv,
            scoring=cv_config["scoring"], n_jobs=cv_config["n_jobs"],
        )
        rows.append({
            "model": candidate["type"],
            "validation_mean": folds["validation"].mean(),
            "validation_std": folds["validation"].std(ddof=1),
            "train_mean": folds["train"].mean(),
            "gap_mean": folds["gap"].mean(),
        })
        fold_reports.append((candidate["type"], folds))

    defaults = pd.DataFrame(rows).sort_values(
        "validation_mean", ascending=False, kind="stable"
    ).reset_index(drop=True)
    default_text = comparison_report(defaults, cv_config["scoring"])

    tuning_config = config.get("tuning", {})
    tuned_text = ""
    selected_tuned_params = {}
    best_pipelines = {}
    if tuning_config.get("enabled", False):
        inner_cv = StratifiedKFold(
            n_splits=tuning_config["n_splits"],
            shuffle=True,
            random_state=tuning_config["random_state"],
        )
        tuned_rows = []
        tuning_details = []
        for candidate in config["models"]:
            name = candidate["type"]
            default_row = defaults.loc[defaults["model"] == name].iloc[0]
            if name == "dummy":
                tuned_rows.append({
                    "model": name,
                    "default_cv_mean": default_row["validation_mean"],
                    "nested_cv_mean": default_row["validation_mean"],
                    "nested_cv_std": default_row["validation_std"],
                    "gap_mean": default_row["gap_mean"],
                    "trials": 0,
                })
                continue

            search_space = tuning_config["search_spaces"].get(name)
            if not search_space:
                raise ValueError(f"No tuning search space configured for {name}")
            n_trials = tuning_config["n_trials"][name]
            pipeline = build_pipeline(config["preprocessing"], candidate)
            print(f"\nTuning {name}: {n_trials} trials, {tuning_config['n_splits']} inner folds, "
                  f"{cv_config['n_splits']} outer folds", flush=True)
            nested, _ = nested_cross_validate(
                pipeline, X_dev, y_dev, cv, inner_cv, cv_config["scoring"],
                search_space, n_trials, tuning_config["random_state"],
                cv_config["n_jobs"],
            )
            best_pipeline, study = tune_pipeline(
                pipeline, X_dev, y_dev, inner_cv, cv_config["scoring"],
                search_space, n_trials, tuning_config["random_state"],
                cv_config["n_jobs"],
            )
            best_pipelines[name] = best_pipeline
            tuned_rows.append({
                "model": name,
                "default_cv_mean": default_row["validation_mean"],
                "nested_cv_mean": nested["validation"].mean(),
                "nested_cv_std": nested["validation"].std(ddof=1),
                "gap_mean": nested["gap"].mean(),
                "trials": n_trials,
            })
            tuning_details.append(
                f"{name} nested fold scores:\n"
                + nested.to_string(index=False, float_format=lambda value: f"{value:.3f}")
                + "\n\n" + tuning_report(study, nested, cv_config["scoring"])
            )

        tuned = pd.DataFrame(tuned_rows).sort_values(
            "nested_cv_mean", ascending=False, kind="stable"
        ).reset_index(drop=True)
        tuned_text = (
            f"Tuned model comparison (nested CV; metric: {cv_config['scoring']})\n\n"
            + tuned.to_string(index=False, float_format=lambda value: f"{value:.3f}")
        )
        print("\n" + tuned_text)
        winner_name = tuned.iloc[0]["model"]
    else:
        winner_name = defaults.iloc[0]["model"]

    winner_config = next(item for item in config["models"] if item["type"] == winner_name)
    selection_method = "nested CV" if tuning_config.get("enabled", False) else "CV"
    print(f"\nSelected by mean {selection_method} {cv_config['scoring']}: {winner_name}")

    # The search on all development rows supplies the settings to keep; outer
    # CV, rather than the best trial's inner score, selected the algorithm.
    model = best_pipelines.get(winner_name)
    if model is None:
        model = build_pipeline(config["preprocessing"], winner_config)
    else:
        selected_tuned_params = {
            key: value for key, value in model.get_params().items()
            if key in tuning_config["search_spaces"][winner_name]
        }
    model.fit(X_dev, y_dev)
    y_train_pred = model.predict(X_dev)
    y_test_pred = model.predict(X_test)

    report = evaluate(y_dev, y_train_pred, y_test, y_test_pred)
    report += "\n" + fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=config["data"]["sensitive_attr"]
    )
    fold_text = "\n\n".join(
        f"{name} fold scores:\n{folds.to_string(index=False, float_format=lambda value: f'{value:.3f}')}"
        for name, folds in fold_reports
    )
    sections = [default_text, fold_text]
    if tuned_text:
        sections.extend([tuned_text, *tuning_details])
    sections.extend([f"Selected model: {winner_name}\nTuned parameters: {selected_tuned_params}", report])
    report = "\n\n".join(sections)

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report, winner_config, selected_tuned_params)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
