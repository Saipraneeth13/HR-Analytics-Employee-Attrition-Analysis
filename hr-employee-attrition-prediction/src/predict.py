"""Batch and single-record inference script for HR Attrition Risk Scoring."""

import argparse
import joblib
import pandas as pd

from src.data_preprocessing import engineer_features, load_and_clean_data


def score_employees(input_csv: str, output_csv: str, model_path: str = "models/attrition_model.joblib") -> pd.DataFrame:
    """Score employees and assign Low / Medium / High attrition risk tiers."""
    model = joblib.load(model_path)
    raw_df = pd.read_csv(input_csv)
    cleaned_df = engineer_features(load_and_clean_data(input_csv))

    features = cleaned_df.drop(columns=["Attrition"], errors="ignore")
    probs = model.predict_proba(features)[:, 1]

    scored = raw_df.copy()
    scored["AttritionProbability"] = probs.round(4)
    scored["RiskTier"] = pd.cut(
        probs,
        bins=[-0.01, 0.35, 0.65, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )
    scored.to_csv(output_csv, index=False)
    print(f"Scored {len(scored)} records -> saved to {output_csv}")
    return scored


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score employee attrition risk.")
    parser.add_argument("--input", default="data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv", help="Path to input CSV")
    parser.add_argument("--output", default="reports/employee_risk_scores.csv", help="Path to output CSV")
    parser.add_argument("--model", default="models/attrition_model.joblib", help="Path to saved model")
    args = parser.parse_args()
    score_employees(args.input, args.output, args.model)
