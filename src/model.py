"""Model construction."""
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor


def oblique_random_forest(n_estimators=100, n_components=10, density="auto",
                          random_state=None, n_jobs=None, **tree_params):
    """Bag trees grown on separately randomized sparse feature projections."""
    tree_params.setdefault("max_features", "sqrt")
    oblique_tree = Pipeline([
        ("project", SparseRandomProjection(n_components=n_components, density=density)),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])
    return BaggingClassifier(
        estimator=oblique_tree, n_estimators=n_estimators,
        random_state=random_state, n_jobs=n_jobs,
    )


def adaboost(max_depth=1, **params):
    """Boost decision trees; depth one gives the usual decision stumps."""
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_depth), **params
    )


def adaboost_logistic(**params):
    """Challenge model: boost logistic-regression classifiers."""
    return AdaBoostClassifier(
        estimator=LogisticRegression(max_iter=2000), **params
    )

_MODELS = {
    # always predicts the most frequent class -- the floor every real model must beat
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    # Bootstrap-averaged trees with random feature subsets at each split.
    "random_forest": RandomForestClassifier,
    "oblique_random_forest": oblique_random_forest,
    "adaboost": adaboost,
    "adaboost_logistic": adaboost_logistic,
}


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params", {})

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


def build_pipeline(preprocessing_config: dict, model_config: dict) -> Pipeline:
    """Keep every learned preprocessing step inside the estimator being evaluated."""
    return Pipeline([
        ("prep", build_preprocessor(preprocessing_config)),
        ("model", build_model(model_config)),
    ])
