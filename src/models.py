"""
Machine Learning Severity Classification Pipeline
Implements:
1. Linear Regression (Threshold-rounded linear classifier)
2. Multinomial Logistic Regression
3. J48 / C4.5 Decision Tree (Entropy criterion)
4. Random Forest Classifier (100 estimators)
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO"]
ORDERED_CLASSES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

class LinearRegressionThresholdClassifier:
    """
    Ordinary Least Squares linear regression model trained on continuous AQI,
    operating as a discrete classifier through regulatory threshold boundaries.
    """
    def __init__(self):
        self.model = LinearRegression()
        self.classes_ = np.array(ORDERED_CLASSES)

    def fit(self, X: np.ndarray, y_continuous: np.ndarray):
        self.model.fit(X, y_continuous)
        return self

    def predict_continuous(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        continuous_preds = self.predict_continuous(X)
        buckets = []
        for val in continuous_preds:
            if val <= 50.0:
                buckets.append("Good")
            elif val <= 100.0:
                buckets.append("Satisfactory")
            elif val <= 200.0:
                buckets.append("Moderate")
            elif val <= 300.0:
                buckets.append("Poor")
            elif val <= 400.0:
                buckets.append("Very Poor")
            else:
                buckets.append("Severe")
        return np.array(buckets)

def train_and_evaluate_models(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains all four requested models with standard scaling on a 75:25 split.
    """
    X = df[FEATURE_COLS].copy()
    y_class = df["AQI_Bucket"].copy()
    y_cont = df["AQI"].copy()

    X_train, X_test, y_train_cls, y_test_cls, y_train_cnt, y_test_cnt = train_test_split(
        X, y_class, y_cont,
        test_size=test_size,
        random_state=random_state,
        stratify=y_class
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression (Threshold Classifier)": LinearRegressionThresholdClassifier(),
        "Logistic Regression (Multinomial)": LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state
        ),
        "J48 / C4.5 Decision Tree": DecisionTreeClassifier(
            criterion="entropy",
            random_state=random_state
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100,
            criterion="entropy",
            random_state=random_state
        )
    }

    trained = {}
    for name, model in models.items():
        if isinstance(model, LinearRegressionThresholdClassifier):
            model.fit(X_train_scaled, y_train_cnt.values)
        else:
            model.fit(X_train_scaled, y_train_cls.values)

        preds = model.predict(X_test_scaled)
        trained[name] = {
            "model": model,
            "predictions": preds
        }

    return {
        "trained_models": trained,
        "scaler": scaler,
        "X_test": X_test,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test_cls,
        "classes": ORDERED_CLASSES,
        "feature_cols": FEATURE_COLS
    }

def score_batch_records(
    df_input: pd.DataFrame,
    model_name: str,
    pipeline_bundle: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes batch inference on custom uploaded CSV.
    Validates required pollutant schema and produces enriched predictions.
    """
    scaler = pipeline_bundle["scaler"]
    model = pipeline_bundle["trained_models"][model_name]["model"]

    missing = [c for c in FEATURE_COLS if c not in df_input.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing mandatory pollutant column(s): {', '.join(missing)}")

    X_raw = df_input[FEATURE_COLS].copy()
    X_filled = X_raw.fillna(X_raw.median())
    X_scaled = scaler.transform(X_filled)

    preds = model.predict(X_scaled)
    output_df = df_input.copy()
    output_df["Predicted_AQI_Bucket"] = preds

    if isinstance(model, LinearRegressionThresholdClassifier):
        output_df["Predicted_AQI_Score"] = model.predict_continuous(X_scaled).round(1)

    metadata = {
        "count": len(output_df),
        "model_used": model_name,
        "distribution": pd.Series(preds).value_counts().to_dict()
    }
    return output_df, metadata
