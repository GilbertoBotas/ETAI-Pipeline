"""Run the two optional Week 6 challenges with a small reproducible budget."""

from copy import deepcopy
from pathlib import Path

import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from main import load_config
from src.data import load_data
from src.evaluate import cross_validate_pipeline
from src.model import build_pipeline
from src.preprocessing import (
    add_engineered_features, clean_dataset, split_dev_test, split_features_target,
)
from src.tuning import nested_cross_validate


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    config = load_config()
    clean = clean_dataset(load_data(config["data"]["path"]), config["diagnostics"])
    engineered = add_engineered_features(clean)
    X, y, extras = split_features_target(
        engineered, config["data"], config["preprocessing"]["mnar_indicator_sources"]
    )
    X_dev, _, y_dev, _, _, _ = split_dev_test(
        X, y, extras, config["test_set"]["size"], config["test_set"]["random_state"]
    )

    outer = StratifiedKFold(3, shuffle=True, random_state=42)
    inner = StratifiedKFold(2, shuffle=True, random_state=42)
    no_features = deepcopy(config["preprocessing"])
    added = set(config["feature_engineering"]["columns"])
    no_features["numeric_features"] = [
        column for column in no_features["numeric_features"] if column not in added
    ]

    rows = []
    for name in ["logistic_regression", "random_forest", "adaboost"]:
        model = next(item for item in config["models"] if item["type"] == name)
        # Keep ensemble sizes modest; both feature conditions use the same model.
        model = deepcopy(model)
        if name in {"random_forest", "adaboost"}:
            model["params"]["n_estimators"] = 30
        for label, prep in [("without", no_features), ("with", config["preprocessing"])]:
            print(f"Challenge 1: {name} {label} engineered features", flush=True)
            folds, _ = nested_cross_validate(
                build_pipeline(prep, model), X_dev, y_dev, outer, inner,
                "accuracy", config["tuning"]["search_spaces"][name],
                4, 42,
            )
            rows.append({
                "model": name,
                "features": label,
                "mean": folds["validation"].mean(),
                "std": folds["validation"].std(ddof=1),
                "fold_scores": ", ".join(f"{score:.3f}" for score in folds["validation"]),
            })

    features_table = pd.DataFrame(rows)
    boosted = {"type": "adaboost_logistic", "params": {
        "n_estimators": 50, "learning_rate": 1.0, "random_state": 42,
    }}
    logistic = next(item for item in config["models"] if item["type"] == "logistic_regression")
    boosted_rows = []
    for label, model in [("logistic_regression", logistic), ("adaboost_logistic", boosted)]:
        folds = cross_validate_pipeline(
            build_pipeline(config["preprocessing"], model), X_dev, y_dev, outer
        )
        boosted_rows.append({
            "model": label,
            "mean": folds["validation"].mean(),
            "std": folds["validation"].std(ddof=1),
            "fold_scores": ", ".join(f"{score:.3f}" for score in folds["validation"]),
        })
    boost_table = pd.DataFrame(boosted_rows)

    output = Path(config["output"]["results_dir"])
    output.mkdir(exist_ok=True)
    features_table.to_csv(output / "week6_feature_challenge.csv", index=False)
    boost_table.to_csv(output / "week6_boosted_logistic.csv", index=False)
    print("\nChallenge 1\n", features_table.to_string(index=False), flush=True)
    print("\nChallenge 2\n", boost_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
