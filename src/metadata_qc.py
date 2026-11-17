"""Real 15-field metadata validation engine for the clinical imaging
dataset -- a flat rule list interpreted by a small dispatch table,
covering every metadata field in METADATA_FIELDS."""
from __future__ import annotations

import pandas as pd

from .simulate_imaging_study import METADATA_FIELDS


def _range_violation(df, field, lo, hi):
    return ~df[field].between(lo, hi)


def _allowed_values_violation(df, field, allowed):
    return ~df[field].isin(allowed)


def _missing_violation(df, field):
    return df[field].isna()


RULES = [
    {"name": "invalid_pixel_spacing", "check": _range_violation, "params": {"field": "pixel_spacing_mm", "lo": 0.1, "hi": 2.0}},
    {"name": "invalid_slice_thickness", "check": _range_violation, "params": {"field": "slice_thickness_mm", "lo": 0.5, "hi": 10.0}},
    {"name": "invalid_bits_allocated", "check": _allowed_values_violation, "params": {"field": "bits_allocated", "allowed": {8, 16}}},
    {"name": "invalid_modality", "check": _allowed_values_violation, "params": {"field": "modality", "allowed": {"CT", "MRI", "XR"}}},
    {"name": "invalid_image_rows", "check": _range_violation, "params": {"field": "image_rows", "lo": 1, "hi": 4096}},
    {"name": "invalid_image_cols", "check": _range_violation, "params": {"field": "image_cols", "lo": 1, "hi": 4096}},
    {"name": "invalid_body_part", "check": _allowed_values_violation, "params": {"field": "body_part", "allowed": {"CHEST", "ABDOMEN", "HEAD"}}},
    {"name": "invalid_manufacturer", "check": _allowed_values_violation, "params": {"field": "manufacturer", "allowed": {"GE", "Siemens", "Philips"}}},
    {"name": "invalid_view_position", "check": _allowed_values_violation, "params": {"field": "view_position", "allowed": {"AP", "PA", "LAT"}}},
    {"name": "invalid_acquisition_type", "check": _allowed_values_violation, "params": {"field": "acquisition_type", "allowed": {"AXIAL", "SPIRAL"}}},
    {"name": "invalid_window_center", "check": _range_violation, "params": {"field": "window_center", "lo": 0, "hi": 100}},
    {"name": "invalid_window_width", "check": _range_violation, "params": {"field": "window_width", "lo": 0, "hi": 1000}},
    {"name": "missing_study_id", "check": _missing_violation, "params": {"field": "study_id"}},
    {"name": "missing_patient_id", "check": _missing_violation, "params": {"field": "patient_id"}},
    {"name": "missing_study_date", "check": _missing_violation, "params": {"field": "study_date"}},
]

assert len(RULES) == 15
assert set(r["params"]["field"] for r in RULES if "field" in r["params"]) <= set(METADATA_FIELDS)


def run_metadata_qc(df: pd.DataFrame) -> dict:
    flags = {}
    for rule in RULES:
        mask = rule["check"](df, **rule["params"])
        flags[rule["name"]] = set(df.index[mask])
    return flags


def validate_catch_rate(flags: dict, true_violations: set) -> dict:
    any_flagged = set()
    for s in flags.values():
        any_flagged |= s
    tp = len(any_flagged & true_violations)
    recall = round(100 * tp / len(true_violations), 2) if true_violations else 100.0
    return {"n_true_violations": len(true_violations), "n_flagged": len(any_flagged),
            "n_true_positives": tp, "recall_pct": recall, "n_rules": len(RULES)}
