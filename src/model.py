"""Model and leak-safe feature-selection pipeline construction."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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


class VoteSelector(SelectorMixin, BaseEstimator):
    """Keep features selected by at least ``min_votes`` of three fitted models."""

    def __init__(self, min_votes=2, random_state=None):
        self.min_votes = min_votes
        self.random_state = random_state

    def fit(self, X, y):
        if self.min_votes not in (1, 2, 3):
            raise ValueError("min_votes must be 1, 2, or 3")
        forest = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=20,
            random_state=self.random_state, n_jobs=-1,
        ).fit(X, y)
        booster = GradientBoostingClassifier(random_state=self.random_state).fit(X, y)
        lasso = LogisticRegression(l1_ratio=1.0, C=0.01, solver="liblinear").fit(X, y)
        self.votes_ = (
            (forest.feature_importances_ > forest.feature_importances_.mean()).astype(int)
            + (booster.feature_importances_ > booster.feature_importances_.mean())
            + (np.abs(lasso.coef_[0]) > 1e-8)
        )
        self.support_ = self.votes_ >= self.min_votes
        if not self.support_.any():
            raise ValueError("Feature selection removed every feature; lower min_votes")
        self.n_features_in_ = X.shape[1]
        return self

    def _get_support_mask(self):
        return self.support_

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
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "naive_bayes": GaussianNB,
    "xgboost": XGBClassifier,
}


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params", {})

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


def build_pipeline(preprocessing_config: dict, model_config: dict,
                   selection_config: dict = None, memory=None) -> Pipeline:
    """Keep every learned preprocessing step inside the estimator being evaluated."""
    steps = [("prep", build_preprocessor(preprocessing_config))]
    if selection_config and selection_config.get("enabled", False):
        steps.append(("select", VoteSelector(
            min_votes=selection_config.get("min_votes", 2),
            random_state=selection_config.get("random_state"),
        )))
    steps.append(("model", build_model(model_config)))
    return Pipeline(steps, memory=memory)
