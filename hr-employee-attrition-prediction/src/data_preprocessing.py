"""Data cleaning and feature engineering module for HR Employee Attrition."""

import pandas as pd


DROP_COLUMNS = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load raw HR CSV data and drop constant/identifier columns."""
    df = pd.read_csv(filepath)
    cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    return df.drop(columns=cols_to_drop).copy()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-driven HR retention and career velocity features."""
    data = df.copy()
    data["IncomePerYearWorked"] = data["MonthlyIncome"] / (data["TotalWorkingYears"] + 1)
    data["TenureRatio"] = data["YearsAtCompany"] / (data["TotalWorkingYears"] + 1)
    data["PromotionLag"] = data["YearsAtCompany"] - data["YearsSinceLastPromotion"]
    data["SatisfactionIndex"] = (
        data["EnvironmentSatisfaction"]
        + data["JobSatisfaction"]
        + data["RelationshipSatisfaction"]
        + data["WorkLifeBalance"]
    ) / 4.0
    data["OverTimeFlag"] = (data["OverTime"] == "Yes").astype(int)
    return data


if __name__ == "__main__":
    raw_df = load_and_clean_data("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    processed_df = engineer_features(raw_df)
    processed_df.to_csv("data/processed/hr_attrition_processed.csv", index=False)
    print(f"Saved processed dataset with shape: {processed_df.shape}")
