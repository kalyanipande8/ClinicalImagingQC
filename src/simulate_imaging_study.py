"""Simulate 900 clinical imaging studies with 15 real metadata fields
(with injected validation defects) plus a matched EHR record and an
image-derived feature, where a clinical outcome is driven by both EHR
and imaging signal -- the ground truth a metadata-QC + multimodal-
prediction pipeline must recover."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_STUDIES = 900
SEED = 983

METADATA_FIELDS = ["study_id", "patient_id", "modality", "body_part", "pixel_spacing_mm",
                    "slice_thickness_mm", "manufacturer", "study_date", "image_rows", "image_cols",
                    "bits_allocated", "window_center", "window_width", "view_position", "acquisition_type"]


def simulate(n_studies: int = N_STUDIES, seed: int = SEED):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "study_id": [f"IMG-{i:05d}" for i in range(n_studies)],
        "patient_id": [f"PT-{i:05d}" for i in range(n_studies)],
        "modality": rng.choice(["CT", "MRI", "XR"], n_studies),
        "body_part": rng.choice(["CHEST", "ABDOMEN", "HEAD"], n_studies),
        "pixel_spacing_mm": np.round(rng.uniform(0.3, 1.2, n_studies), 3),
        "slice_thickness_mm": np.round(rng.uniform(1.0, 5.0, n_studies), 2),
        "manufacturer": rng.choice(["GE", "Siemens", "Philips"], n_studies),
        "study_date": pd.date_range("2024-01-01", periods=n_studies, freq="h"),
        "image_rows": rng.choice([256, 512, 1024], n_studies),
        "image_cols": rng.choice([256, 512, 1024], n_studies),
        "bits_allocated": rng.choice([8, 16], n_studies),
        "window_center": rng.integers(30, 60, n_studies),
        "window_width": rng.integers(300, 500, n_studies),
        "view_position": rng.choice(["AP", "PA", "LAT"], n_studies),
        "acquisition_type": rng.choice(["AXIAL", "SPIRAL"], n_studies),
    })

    # real EHR features + imaging feature driving the true outcome
    age = rng.integers(20, 85, n_studies)
    comorbidity_score = rng.poisson(1.5, n_studies)
    imaging_feature = rng.normal(0, 1, n_studies)  # e.g. a radiomics-style intensity feature

    logit = 0.04 * (age - 50) + 0.5 * comorbidity_score + 1.3 * imaging_feature + rng.normal(0, 1.0, n_studies)
    outcome_prob = 1 / (1 + np.exp(-logit))
    outcome = rng.binomial(1, outcome_prob)

    df["age"] = age
    df["comorbidity_score"] = comorbidity_score
    df["imaging_feature"] = np.round(imaging_feature, 4)
    df["adverse_outcome"] = outcome

    # inject real metadata defects across several of the 15 fields
    true_violations = set()
    idx = rng.choice(df.index, int(n_studies * 0.02), replace=False)
    df.loc[idx, "pixel_spacing_mm"] = -1.0
    true_violations |= set(idx)
    idx = rng.choice(df.index, int(n_studies * 0.015), replace=False)
    df.loc[idx, "bits_allocated"] = 99
    true_violations |= set(idx)
    idx = rng.choice(df.index, int(n_studies * 0.015), replace=False)
    df.loc[idx, "modality"] = "UNKNOWN"
    true_violations |= set(idx)
    idx = rng.choice(df.index, int(n_studies * 0.01), replace=False)
    df.loc[idx, "image_rows"] = 0
    true_violations |= set(idx)

    return df, true_violations


if __name__ == "__main__":
    df, violations = simulate()
    print(df.shape, len(violations))
