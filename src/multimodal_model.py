"""Real cross-validated comparison between an EHR-only baseline
classifier and a multimodal classifier that adds the imaging-derived
feature -- the real test of whether integrating imaging features with
EHR data genuinely improves prediction, not an asserted percentage."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score


def evaluate_ehr_only(df: pd.DataFrame, cv: int = 5, seed: int = 983) -> float:
    X = df[["age", "comorbidity_score"]].values
    y = df["adverse_outcome"].values
    model = GradientBoostingClassifier(random_state=seed, n_estimators=100, max_depth=3)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return float(scores.mean())


def evaluate_multimodal(df: pd.DataFrame, cv: int = 5, seed: int = 983) -> float:
    X = df[["age", "comorbidity_score", "imaging_feature"]].values
    y = df["adverse_outcome"].values
    model = GradientBoostingClassifier(random_state=seed, n_estimators=100, max_depth=3)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return float(scores.mean())


def compare_models(df: pd.DataFrame) -> dict:
    ehr_only_acc = evaluate_ehr_only(df)
    multimodal_acc = evaluate_multimodal(df)
    improvement_pct = round(100 * (multimodal_acc - ehr_only_acc) / ehr_only_acc, 2)
    return {"ehr_only_accuracy": round(ehr_only_acc, 4), "multimodal_accuracy": round(multimodal_acc, 4),
            "relative_improvement_pct": improvement_pct}
