"""
Entry point for model comparison and final evaluation.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load -> clean -> split -> cross-validate candidates on development rows
    -> refit the winner -> evaluate once on held-out rows -> save results
"""
import yaml
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from src.data import load_data
from src.preprocessing import clean_dataset, split_features_target, build_preprocessor, split_dev_test
from src.model import build_model
from src.evaluate import cross_validate_pipeline, comparison_report, evaluate, fairness_report
from src.results import save_run
import pandas as pd



def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

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

    def make_pipeline(model_config):
        return Pipeline([
            ("prep", build_preprocessor(config["preprocessing"])),
            ("model", build_model(model_config)),
        ])

    rows = []
    fold_reports = []
    for candidate in config["models"]:
        folds = cross_validate_pipeline(
            make_pipeline(candidate), X_dev, y_dev, cv,
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

    comparisons = pd.DataFrame(rows).sort_values(
        "validation_mean", ascending=False, kind="stable"
    ).reset_index(drop=True)
    comparison_text = comparison_report(comparisons, cv_config["scoring"])
    winner_name = comparisons.iloc[0]["model"]
    winner_config = next(item for item in config["models"] if item["type"] == winner_name)
    print(f"\nSelected by mean CV {cv_config['scoring']}: {winner_name}")

    # CV estimates the recipe's performance; refit a fresh model on all
    # development rows before the one held-out test evaluation.
    model = make_pipeline(winner_config)
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
    report = f"{comparison_text}\n\nSelected model: {winner_name}\n\n{fold_text}\n\n{report}"

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report, winner_config)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
