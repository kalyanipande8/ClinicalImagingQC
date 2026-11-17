import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulate_imaging_study import simulate
from src.metadata_qc import run_metadata_qc, validate_catch_rate, RULES
from src.multimodal_model import compare_models


def test_15_metadata_rules_defined():
    assert len(RULES) == 15


def test_qc_catches_most_violations():
    df, true_violations = simulate()
    flags = run_metadata_qc(df)
    catch_rate = validate_catch_rate(flags, true_violations)
    assert catch_rate["recall_pct"] >= 90.0


def test_multimodal_beats_ehr_only():
    df, _ = simulate()
    result = compare_models(df)
    assert result["multimodal_accuracy"] > result["ehr_only_accuracy"]


def test_relative_improvement_is_positive():
    df, _ = simulate()
    result = compare_models(df)
    assert result["relative_improvement_pct"] > 0


def test_study_count_is_900():
    df, _ = simulate()
    assert len(df) == 900
