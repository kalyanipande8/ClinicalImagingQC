import json
import os

from src.simulate_imaging_study import simulate
from src.metadata_qc import run_metadata_qc, validate_catch_rate
from src.multimodal_model import compare_models


def main():
    df, true_violations = simulate()
    flags = run_metadata_qc(df)
    catch_rate = validate_catch_rate(flags, true_violations)
    model_comparison = compare_models(df)

    summary = {"n_studies": len(df), "metadata_qc": catch_rate, "multimodal_comparison": model_comparison}

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
