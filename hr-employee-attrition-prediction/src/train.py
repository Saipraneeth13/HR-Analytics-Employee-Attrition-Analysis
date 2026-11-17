"""Model training, cross-validation, and evaluation pipeline."""

import json
import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_preprocessing import engineer_features, load_and_clean_data


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build ColumnTransformer for numeric scaling and categorical one-hot encoding."""
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ]
    )


def train_and_evaluate(raw_path: str = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv") -> None:
    """Run end-to-end training, model comparison, and artifact serialization."""
    df = engineer_features(load_and_clean_data(raw_path))
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/hr_attrition_processed.csv", index=False)

    X = df.drop(columns=["Attrition"])
    y = (df["Attrition"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X)
    candidates = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_model = None
    best_auc = -1.0

    for name, clf in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        pipe.fit(X_train, y_train)

        probs = pipe.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        test_auc = roc_auc_score(y_test, probs)

        results[name] = {
            "CV_ROC_AUC": round(float(cv_auc), 4),
            "Test_ROC_AUC": round(float(test_auc), 4),
            "PR_AUC": round(float(average_precision_score(y_test, probs)), 4),
            "Recall": round(float(recall_score(y_test, preds)), 4),
            "Precision": round(float(precision_score(y_test, preds)), 4),
            "F1": round(float(f1_score(y_test, preds)), 4),
        }
        print(f"=== {name} (Test ROC-AUC: {test_auc:.4f}) ===")
        print(classification_report(y_test, preds))

        if test_auc > best_auc:
            best_auc = test_auc
            best_model = pipe

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    joblib.dump(best_model, "models/attrition_model.joblib")
    with open("reports/model_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved best model to models/attrition_model.joblib")


if __name__ == "__main__":
    train_and_evaluate()
