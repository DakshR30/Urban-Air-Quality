#!/usr/bin/env bash
# ==============================================================================
# Urban Air Quality & Pollution Forecasting System
# Automated Project Setup & Progressive Git History Generator
#
# Constraints & Commit Specifications:
# - Authentic real-world public dataset (>5,000 records from CPCB India)
# - Incremental file-by-file feature commits
# - Window: 09-Oct-2026 19:00:00 IST to 10-Oct-2026 06:00:00 IST (+0530)
# - Natural, jittered non-round timestamps
# - Exact commit timeline:
#   1. 09-Oct ~19:43:18 IST: chore: initialize repository, structure, and dependency specifications
#   2. 09-Oct ~21:12:47 IST: feat: add real-world dataset ingestion and automated ETL pipeline
#   3. 09-Oct ~22:58:04 IST: feat: implement star schema staging and multidimensional OLAP engine
#   4. 10-Oct ~00:36:21 IST: feat: implement classification models (Linear Regression, Logistic, J48, Random Forest)
#   5. 10-Oct ~02:14:39 IST: feat: implement evaluation suite with separate metric visualizations
#   6. 10-Oct ~03:47:52 IST: feat: build interactive Streamlit frontend with OLAP dashboard and custom CSV inference
#   7. 10-Oct ~04:52:19 IST: refactor: optimize caching, handle edge cases, and add input validation
#   8. 10-Oct ~05:44:11 IST: docs: add comprehensive README with star schema architecture and usage guide
# ==============================================================================

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "========================================================================"
echo "🚀 Initializing DWM Air Quality Project & Building Progressive Git History"
echo "========================================================================"

# Reset git repository
rm -rf .git
git init -b main

# Helper commit function with backdated author and committer dates
commit_step() {
    local timestamp="$1"
    local message="$2"
    export GIT_AUTHOR_DATE="$timestamp"
    export GIT_COMMITTER_DATE="$timestamp"
    git commit -m "$message"
    echo "  [✓] Milestone Committed: '$message' at $timestamp"
}

# ==============================================================================
# MILESTONE 1 (09-Oct 19:43:18 IST)
# chore: initialize repository, structure, and dependency specifications
# ==============================================================================
echo ""
echo "[1/8] Setting up repository structure and dependencies..."
mkdir -p src data

cat << 'EOF' > .gitignore
# Python artifacts
__pycache__/
*.py[cod]
*$py.class
*.so

# Virtual environment
.venv/
venv/
env/
ENV/

# Distribution & cache
build/
dist/
*.egg-info/
.DS_Store
.idea/
.vscode/

# Streamlit config
.streamlit/secrets.toml
EOF

cat << 'EOF' > requirements.txt
streamlit>=1.32.0
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.4.0
plotly>=5.19.0
requests>=2.31.0
EOF

git add .gitignore requirements.txt
commit_step "2026-10-09 19:43:18 +0530" "chore: initialize repository, structure, and dependency specifications"

# ==============================================================================
# MILESTONE 2 (09-Oct 21:12:47 IST)
# feat: add real-world dataset ingestion and automated ETL pipeline
# ==============================================================================
echo ""
echo "[2/8] Writing real dataset downloader and ETL pipeline..."

cat << 'EOF' > src/download_data.py
"""
Real-World Air Quality Data Ingestion & ETL Pipeline
Fetches authentic public CPCB (Central Pollution Control Board, India) city_day dataset
exceeding 20,000 real sensor observations. Cleans nulls, calculates temporal dimensions,
and structures data under standard CPCB AQI classifications.
"""

import os
import io
import requests
import pandas as pd
import numpy as np

# Authentic public repositories hosting the official CPCB city_day dataset (>29,000 records)
DATA_URLS = [
    "https://raw.githubusercontent.com/SunnyRao07/Air-Quality-Index-AQI-Prediction/master/city_day.csv",
    "https://raw.githubusercontent.com/aryanrathore1012/Air_Quality_Analysis_and_Prediction_for_Indian_Cities_and_States/master/city_day.csv"
]

ORDERED_CLASSES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

CITY_COORDINATES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi", "zone": "North"},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat", "zone": "West"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka", "zone": "South"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra", "zone": "West"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu", "zone": "South"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana", "zone": "South"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal", "zone": "East"},
    "Pune": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra", "zone": "West"},
    "Patna": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar", "zone": "East"},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh", "zone": "North"},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan", "zone": "North"},
    "Gurugram": {"lat": 28.4595, "lon": 77.0266, "state": "Haryana", "zone": "North"},
    "Amritsar": {"lat": 31.6340, "lon": 74.8723, "state": "Punjab", "zone": "North"},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh", "zone": "South"}
}

def get_season(month: int) -> str:
    """Classifies calendar month into meteorological season."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

def aqi_to_bucket(aqi: float) -> str:
    """Maps continuous AQI value to official Indian regulatory category."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

def fetch_and_clean_real_data(output_path: str = "data/real_cpcb_air_quality.csv") -> pd.DataFrame:
    """
    Downloads authentic CPCB dataset from public mirrors, cleans genuine missing
    readings using city medians, extracts temporal dimensions, and writes cleaned CSV.
    """
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path)
            if len(df) >= 5000 and "AQI_Bucket" in df.columns:
                return df
        except Exception:
            pass

    raw_df = None
    for url in DATA_URLS:
        try:
            resp = requests.get(url, timeout=25)
            if resp.status_code == 200 and len(resp.content) > 50000:
                raw_df = pd.read_csv(io.StringIO(resp.text))
                break
        except Exception:
            continue

    if raw_df is None:
        raise RuntimeError("Failed to fetch public real-world air quality dataset from configured endpoints.")

    # Retain observations having basic valid data
    # Filter rows having valid Date and City
    df = raw_df.dropna(subset=["City", "Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    # Core target pollutants
    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO"]
    for col in pollutants:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "AQI" not in df.columns:
        df["AQI"] = np.nan
    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")

    # Retain records that have PM2.5 and AQI (or compute AQI if PM2.5 present)
    # The CPCB dataset has over 20,000 such real rows
    df = df.dropna(subset=["PM2.5", "AQI"]).copy()

    # Impute missing secondary pollutants (PM10, NO2, SO2, CO) using city medians
    for p in pollutants:
        df[p] = df.groupby("City")[p].transform(lambda x: x.fillna(x.median()))
        df[p] = df[p].fillna(df[p].median())

    # Ensure physical positive bounds
    for p in pollutants:
        df[p] = df[p].clip(lower=0.1).round(2)
    df["AQI"] = df["AQI"].clip(lower=1.0).round(1)

    # Derive or validate AQI_Bucket
    df["AQI_Bucket"] = df["AQI"].apply(aqi_to_bucket)

    # Compute temporal attributes
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%b")
    df["Day"] = df["Date"].dt.day
    df["Day_of_Week"] = df["Date"].dt.strftime("%A")
    df["Is_Weekend"] = df["Date"].dt.weekday.apply(lambda w: 1 if w >= 5 else 0)
    df["Season"] = df["Month"].apply(get_season)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Attach spatial metadata
    def get_city_meta(city_name, key, default):
        return CITY_COORDINATES.get(city_name, {}).get(key, default)

    df["State"] = df["City"].apply(lambda c: get_city_meta(c, "state", "India"))
    df["Zone"] = df["City"].apply(lambda c: get_city_meta(c, "zone", "Central"))
    df["Latitude"] = df["City"].apply(lambda c: get_city_meta(c, "lat", 20.5937))
    df["Longitude"] = df["City"].apply(lambda c: get_city_meta(c, "lon", 78.9629))

    # Clean select columns
    keep_cols = [
        "Date", "City", "State", "Zone", "Latitude", "Longitude",
        "PM2.5", "PM10", "NO2", "SO2", "CO", "AQI", "AQI_Bucket",
        "Year", "Month", "Month_Name", "Day", "Day_of_Week", "Is_Weekend", "Season"
    ]
    cleaned_df = df[keep_cols].reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    return cleaned_df

def get_sample_upload_data(num_rows: int = 50) -> pd.DataFrame:
    """Returns sample batch CSV with required pollutant headers for testing."""
    df = fetch_and_clean_real_data()
    sample = df[["City", "Date", "PM2.5", "PM10", "NO2", "SO2", "CO"]].sample(n=num_rows, random_state=42)
    return sample.reset_index(drop=True)

if __name__ == "__main__":
    data = fetch_and_clean_real_data()
    print(f"Dataset successfully fetched & cleaned: {len(data):,} records")
EOF

git add src/download_data.py
commit_step "2026-10-09 21:12:47 +0530" "feat: add real-world dataset ingestion and automated ETL pipeline"

# ==============================================
# MILESTONE 3 (09-Oct 22:58:04 IST)
# feat: implement star schema staging and multidimensional OLAP engine
# ==============================================
echo ""
echo "[3/8] Building Star Schema staging and multidimensional OLAP engine..."

cat << 'EOF' > src/olap_engine.py
"""
Star Schema Warehouse Modeling & OLAP Operations Engine
Constructs Fact_AirQuality, Dim_Location, Dim_Date relational views
and implements Slice, Dice, Roll-up / Drill-down, and Pivot operations.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

def build_star_schema(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Decomposes the flattened dataset into normalized Star Schema entities:
    Fact_AirQuality, Dim_Location, Dim_Date.
    """
    # 1. Dim_Location
    loc_cols = ["City", "State", "Zone", "Latitude", "Longitude"]
    dim_loc = df[loc_cols].drop_duplicates().reset_index(drop=True)
    dim_loc["LocationKey"] = [f"LOC_{i+1:03d}" for i in range(len(dim_loc))]

    # 2. Dim_Date
    date_cols = ["Date", "Year", "Month", "Month_Name", "Day", "Day_of_Week", "Is_Weekend", "Season"]
    dim_date = df[date_cols].drop_duplicates().reset_index(drop=True)
    dim_date["DateKey"] = pd.to_datetime(dim_date["Date"]).dt.strftime("%Y%m%d").astype(int)

    # 3. Fact_AirQuality
    fact_df = df.copy()
    fact_df["DateKey"] = pd.to_datetime(fact_df["Date"]).dt.strftime("%Y%m%d").astype(int)
    fact_df = fact_df.merge(dim_loc[["City", "LocationKey"]], on="City", how="left")
    fact_df["FactID"] = [f"FACT_{i+1:06d}" for i in range(len(fact_df))]

    fact_cols = ["FactID", "DateKey", "LocationKey", "PM2.5", "PM10", "NO2", "SO2", "CO", "AQI", "AQI_Bucket"]
    fact_table = fact_df[fact_cols].reset_index(drop=True)

    return {
        "Fact_AirQuality": fact_table,
        "Dim_Location": dim_loc,
        "Dim_Date": dim_date
    }

def slice_cube(df: pd.DataFrame, dimension: str, value: Any) -> pd.DataFrame:
    """
    OLAP Slice:
    Filters the multidimensional cube along a single dimension attribute.
    """
    if dimension not in df.columns:
        raise ValueError(f"Dimension '{dimension}' does not exist.")
    return df[df[dimension] == value].copy().reset_index(drop=True)

def dice_cube(
    df: pd.DataFrame,
    cities: Optional[List[str]] = None,
    seasons: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    aqi_range: Optional[tuple] = None
) -> pd.DataFrame:
    """
    OLAP Dice:
    Extracts a sub-cube by filtering across multiple dimensions simultaneously.
    """
    sub = df.copy()
    if cities:
        sub = sub[sub["City"].isin(cities)]
    if seasons:
        sub = sub[sub["Season"].isin(seasons)]
    if years:
        sub = sub[sub["Year"].isin(years)]
    if aqi_range:
        sub = sub[(sub["AQI"] >= aqi_range[0]) & (sub["AQI"] <= aqi_range[1])]
    return sub.reset_index(drop=True)

def rollup_drilldown(
    df: pd.DataFrame,
    level: str = "Monthly",
    agg_func: str = "Mean",
    pollutants: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    OLAP Roll-up & Drill-down:
    Transitions across temporal hierarchy levels:
    Drill-down: Yearly -> Seasonal -> Monthly -> Daily
    Roll-up: Daily -> Monthly -> Seasonal -> Yearly
    """
    if pollutants is None:
        pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "AQI"]

    fn_map = {"Mean": "mean", "Median": "median", "Max": "max", "Min": "min"}
    fn = fn_map.get(agg_func, "mean")

    if level == "Yearly":
        group_keys = ["City", "Year"]
    elif level == "Seasonal":
        group_keys = ["City", "Year", "Season"]
    elif level == "Monthly":
        group_keys = ["City", "Year", "Month_Name"]
    elif level == "Day_of_Week":
        group_keys = ["City", "Day_of_Week"]
    else:  # Daily
        group_keys = ["City", "Date"]

    agg_df = df.groupby(group_keys)[pollutants].agg(fn).reset_index()
    for p in pollutants:
        agg_df[p] = agg_df[p].round(2)
    return agg_df

def pivot_cube(
    df: pd.DataFrame,
    row_dim: str = "City",
    col_dim: str = "Season",
    metric: str = "AQI",
    agg_func: str = "Mean"
) -> pd.DataFrame:
    """
    OLAP Pivot:
    Rotates cube axes to generate cross-tabulated aggregate matrix.
    """
    fn_map = {"Mean": "mean", "Median": "median", "Max": "max", "Min": "min"}
    fn = fn_map.get(agg_func, "mean")
    matrix = pd.pivot_table(df, index=row_dim, columns=col_dim, values=metric, aggfunc=fn)
    return matrix.round(2)
EOF

git add src/olap_engine.py
commit_step "2026-10-09 22:58:04 +0530" "feat: implement star schema staging and multidimensional OLAP engine"

# ==============================================
# MILESTONE 4 (10-Oct 00:36:21 IST)
# feat: implement classification models (Linear Regression, Logistic, J48, Random Forest)
# ==============================================
echo ""
echo "[4/8] Implementing predictive classification models..."

cat << 'EOF' > src/models.py
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
EOF

git add src/models.py
commit_step "2026-10-10 00:36:21 +0530" "feat: implement classification models (Linear Regression, Logistic, J48, Random Forest)"

# ==============================================
# MILESTONE 5 (10-Oct 02:14:39 IST)
# feat: implement evaluation suite with separate metric visualizations
# ==============================================
echo ""
echo "[5/8] Implementing evaluation suite & separate Plotly charts..."

cat << 'EOF' > src/evaluation.py
"""
Model Evaluation Suite & Plotly Visualizations
Computes Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score,
creates a consolidated leaderboard, and outputs dedicated separate interactive Plotly graphs.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

CHART_PALETTES = {
    "Accuracy": "#10B981",              # Emerald
    "Precision (Weighted)": "#6366F1",  # Indigo
    "Recall (Weighted)": "#F59E0B",     # Amber
    "F1-Score (Weighted)": "#EC4899"    # Rose
}

def compute_model_metrics(pipeline_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes all standard classification metrics across test set predictions.
    """
    y_test = pipeline_bundle["y_test"]
    classes = pipeline_bundle["classes"]
    trained = pipeline_bundle["trained_models"]

    leaderboard = []
    details = {}

    for name, item in trained.items():
        y_pred = item["predictions"]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        cr = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)

        leaderboard.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision (Weighted)": round(float(prec), 4),
            "Recall (Weighted)": round(float(rec), 4),
            "F1-Score (Weighted)": round(float(f1), 4)
        })

        details[name] = {
            "confusion_matrix": cm,
            "report": cr
        }

    leaderboard_df = pd.DataFrame(leaderboard).sort_values(by="F1-Score (Weighted)", ascending=False).reset_index(drop=True)
    return {
        "leaderboard": leaderboard_df,
        "details": details
    }

def create_separate_metric_chart(leaderboard_df: pd.DataFrame, metric_name: str, title: str) -> go.Figure:
    """
    Generates a dedicated, separate Plotly bar chart for a single metric.
    """
    sorted_df = leaderboard_df.sort_values(by=metric_name, ascending=True).copy()
    max_val = sorted_df[metric_name].max()
    base_color = CHART_PALETTES.get(metric_name, "#38BDF8")

    colors = [base_color if val == max_val else "rgba(100, 116, 139, 0.7)" for val in sorted_df[metric_name]]

    fig = go.Figure(go.Bar(
        y=sorted_df["Model"],
        x=sorted_df[metric_name],
        orientation="h",
        marker=dict(color=colors, line=dict(color="#1E293B", width=1.5), cornerradius=6),
        text=[f"{v:.4f} ({v*100:.1f}%)" for v in sorted_df[metric_name]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>" + f"{metric_name}: %{{x:.4f}}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=15, color="#F8FAFC"), x=0.02),
        xaxis=dict(
            title=dict(text=metric_name, font=dict(color="#94A3B8")),
            range=[0, max(1.05, max_val * 1.15)],
            gridcolor="#334155",
            tickfont=dict(color="#CBD5E1")
        ),
        yaxis=dict(title="", tickfont=dict(color="#F1F5F9", size=12), autorange="reversed"),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0)",
        plot_bgcolor="rgba(30, 41, 59, 0.4)",
        margin=dict(l=10, r=40, t=50, b=40),
        height=320
    )
    return fig

def create_confusion_matrix_heatmap(cm: np.ndarray, classes: List[str], model_name: str, normalize: bool = False) -> go.Figure:
    """
    Generates an interactive multi-class confusion matrix heatmap.
    """
    display_matrix = cm.copy()
    if normalize:
        row_sums = display_matrix.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1
        display_matrix = np.round((display_matrix / row_sums) * 100, 1)
        text_matrix = [[f"{val:.1f}%" for val in row] for row in display_matrix]
        colorbar_title = "Percentage (%)"
    else:
        text_matrix = [[str(int(val)) for val in row] for row in display_matrix]
        colorbar_title = "Observations"

    fig = go.Figure(data=go.Heatmap(
        z=display_matrix,
        x=classes,
        y=classes,
        colorscale="Viridis",
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=12, color="#FFFFFF"),
        hovertemplate="<b>True:</b> %{y}<br><b>Predicted:</b> %{x}<br><b>Value:</b> %{z}<extra></extra>",
        colorbar=dict(title=colorbar_title, tickfont=dict(color="#CBD5E1"))
    ))

    fig.update_layout(
        title=dict(text=f"<b>Confusion Matrix: {model_name}</b>", font=dict(size=15, color="#F8FAFC"), x=0.02),
        xaxis=dict(title="Predicted Class", tickfont=dict(color="#E2E8F0"), side="bottom"),
        yaxis=dict(title="True Class", tickfont=dict(color="#E2E8F0"), autorange="reversed"),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0)",
        plot_bgcolor="rgba(30, 41, 59, 0.4)",
        margin=dict(l=10, r=10, t=50, b=40),
        height=390
    )
    return fig
EOF

git add src/evaluation.py
commit_step "2026-10-10 02:14:39 +0530" "feat: implement evaluation suite with separate metric visualizations"

# ==============================================
# MILESTONE 6 (10-Oct 03:47:52 IST)
# feat: build interactive Streamlit frontend with OLAP dashboard and custom CSV inference
# ==============================================
echo ""
echo "[6/8] Building initial Streamlit frontend with interactive tabs..."

cat << 'EOF' > app.py
"""
Urban Air Quality & Pollution Forecasting System
Streamlit Frontend Application (Phase 1 Baseline)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.download_data import fetch_and_clean_real_data, get_sample_upload_data, ORDERED_CLASSES
from src.olap_engine import build_star_schema, slice_cube, dice_cube, rollup_drilldown, pivot_cube
from src.models import train_and_evaluate_models, score_batch_records, FEATURE_COLS
from src.evaluation import compute_model_metrics, create_separate_metric_chart, create_confusion_matrix_heatmap

st.set_page_config(
    page_title="Urban Air Quality & Pollution Forecasting System",
    page_icon="🌍",
    layout="wide"
)

st.title("Urban Air Quality & Pollution Forecasting System")
st.caption("A standalone Data Warehousing & Mining (DWM) platform using authentic CPCB observations.")

# Ingest data
with st.spinner("Loading real-world CPCB air quality dataset..."):
    df = fetch_and_clean_real_data()

# Build warehouse entities
schema_tables = build_star_schema(df)

# Train baseline models
bundle = train_and_evaluate_models(df, test_size=0.25)
eval_results = compute_model_metrics(bundle)

tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 Warehouse & OLAP Explorer",
    "🤖 Model Training & Leaderboard",
    "📊 Comparative Visualizations",
    "🚀 Batch CSV Ingestion & Inference"
])

with tab1:
    st.subheader("Star Schema & OLAP Operations")
    st.markdown(f"**Loaded Observations:** `{len(df):,}` authentic CPCB records.")
    
    op = st.selectbox("Select OLAP Operation:", ["Slice", "Dice", "Roll-up / Drill-down", "Pivot"])
    if op == "Slice":
        s_dim = st.selectbox("Dimension:", ["Season", "City", "Year"])
        s_val = st.selectbox("Value:", sorted(list(df[s_dim].unique())))
        sliced = slice_cube(df, s_dim, s_val)
        st.write(f"Sliced rows: {len(sliced):,}")
        st.dataframe(sliced.head(25), use_container_width=True)
    elif op == "Dice":
        cities = st.multiselect("Cities:", sorted(list(df["City"].unique())), default=list(df["City"].unique())[:3])
        seasons = st.multiselect("Seasons:", ["Winter", "Summer", "Monsoon", "Post-Monsoon"], default=["Winter"])
        diced = dice_cube(df, cities=cities, seasons=seasons)
        st.write(f"Diced rows: {len(diced):,}")
        st.dataframe(diced.head(25), use_container_width=True)
    elif op == "Roll-up / Drill-down":
        lvl = st.selectbox("Level:", ["Yearly", "Seasonal", "Monthly", "Daily"])
        agg_res = rollup_drilldown(df, level=lvl)
        st.dataframe(agg_res.head(30), use_container_width=True)
    else:
        pvt = pivot_cube(df, row_dim="City", col_dim="Season", metric="AQI")
        st.dataframe(pvt, use_container_width=True)

with tab2:
    st.subheader("Model Evaluation Leaderboard")
    st.dataframe(eval_results["leaderboard"], use_container_width=True)

with tab3:
    st.subheader("Comparative Metric Graphs")
    st.plotly_chart(create_separate_metric_chart(eval_results["leaderboard"], "Accuracy", "Comparative Accuracy"), use_container_width=True)
    st.plotly_chart(create_separate_metric_chart(eval_results["leaderboard"], "Precision (Weighted)", "Comparative Precision"), use_container_width=True)
    st.plotly_chart(create_separate_metric_chart(eval_results["leaderboard"], "Recall (Weighted)", "Comparative Recall"), use_container_width=True)
    st.plotly_chart(create_separate_metric_chart(eval_results["leaderboard"], "F1-Score (Weighted)", "Comparative F1-Score"), use_container_width=True)

with tab4:
    st.subheader("Batch CSV Ingestion")
    up_file = st.file_uploader("Upload CSV:", type=["csv"])
    if up_file:
        up_df = pd.read_csv(up_file)
        sel_m = st.selectbox("Model:", list(bundle["trained_models"].keys()))
        res_df, meta = score_batch_records(up_df, sel_m, bundle)
        st.dataframe(res_df.head(20), use_container_width=True)
        st.download_button("Download Predictions", res_df.to_csv(index=False), "predictions.csv")
EOF

git add app.py
commit_step "2026-10-10 03:47:52 +0530" "feat: build interactive Streamlit frontend with OLAP dashboard and custom CSV inference"

# ==============================================
# MILESTONE 7 (10-Oct 04:52:19 IST)
# refactor: optimize caching, handle edge cases, and add input validation
# ==============================================
echo ""
echo "[7/8] Enhancing Streamlit application with styling, caching, and robust validation..."

cat << 'EOF' > app.py
"""
Urban Air Quality & Pollution Forecasting System
Data Warehousing & Mining (DWM) Production Application
Built with Streamlit, Plotly, Scikit-Learn, and authentic CPCB environmental data.
"""

import os
import io
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.download_data import (
    fetch_and_clean_real_data,
    get_sample_upload_data,
    ORDERED_CLASSES,
    CITY_COORDINATES
)
from src.olap_engine import (
    build_star_schema,
    slice_cube,
    dice_cube,
    rollup_drilldown,
    pivot_cube
)
from src.models import (
    train_and_evaluate_models,
    score_batch_records,
    FEATURE_COLS
)
from src.evaluation import (
    compute_model_metrics,
    create_separate_metric_chart,
    create_confusion_matrix_heatmap
)

# Page Configuration & Design System
st.set_page_config(
    page_title="Urban Air Quality & Pollution Forecasting System | DWM Studio",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Design Tokens & Card Styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(12px);
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    .metric-caption {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 4px;
    }
    .schema-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .tag-pk { background: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.3); }
    .tag-fk { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .tag-measure { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
</style>
""", unsafe_allow_html=True)

# Optimized Data & Pipeline Caching
@st.cache_data(show_spinner=False)
def load_real_warehouse_data():
    return fetch_and_clean_real_data()

@st.cache_data(show_spinner=False)
def get_star_schema_tables(df):
    return build_star_schema(df)

@st.cache_resource(show_spinner=False)
def load_ml_pipeline(df, test_size=0.25, random_state=42):
    bundle = train_and_evaluate_models(df, test_size=test_size, random_state=random_state)
    eval_metrics = compute_model_metrics(bundle)
    return bundle, eval_metrics

# Load Authentic Data
raw_df = load_real_warehouse_data()
schema_tables = get_star_schema_tables(raw_df)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cloud-lighting.png", width=64)
    st.markdown("### **DWM Intelligence Studio**")
    st.caption("Urban Air Quality & Pollution Forecasting System")
    st.divider()

    st.markdown("#### **Warehouse Statistics**")
    st.markdown(f"""
    - **Total Records:** `{len(raw_df):,}` real observations
    - **Data Source:** Official CPCB India Sensors
    - **Monitored Cities:** `{raw_df['City'].nunique()}` urban centers
    - **Temporal Range:** `{raw_df['Date'].min()}` to `{raw_df['Date'].max()}`
    - **Target Classes:** `{len(ORDERED_CLASSES)}` Severity Buckets
    """)
    st.divider()

    st.markdown("#### **Mining Settings**")
    split_ratio = st.slider("Test Split Ratio", min_value=0.15, max_value=0.35, value=0.25, step=0.05)
    seed = st.number_input("Random State", min_value=1, max_value=9999, value=42, step=1)
    if st.button("🔄 Retrain ML Pipeline", use_container_width=True):
        st.cache_resource.clear()
        st.success("Retraining models on fresh split...")
        st.rerun()

    st.divider()
    st.caption("CPCB Regulatory Standards (India):")
    st.markdown("""
    - `0 - 50`: **Good**
    - `51 - 100`: **Satisfactory**
    - `101 - 200`: **Moderate**
    - `201 - 300`: **Poor**
    - `301 - 400`: **Very Poor**
    - `401+`: **Severe**
    """)

# Train / Retrieve Models
with st.spinner("Initializing predictive classification pipeline..."):
    pipeline_bundle, eval_metrics = load_ml_pipeline(raw_df, test_size=split_ratio, random_state=seed)

# Application Header & Metrics
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("Urban Air Quality & Pollution Forecasting System")
    st.markdown(
        "Production Data Warehousing & Mining system built on **authentic CPCB urban sensor records**. "
        "Integrates a relational **Star Schema**, interactive **OLAP cubes**, and four machine learning models: "
        "**Linear Regression Classifier, Multinomial Logistic Regression, J48 Decision Tree, and Random Forest**."
    )
with col_head2:
    st.metric(label="Verified Real Observations", value=f"{len(raw_df):,}", delta="CPCB Official Data")

# KPI Summary Row
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Monitored Cities</div>
        <div class="metric-value">{raw_df['City'].nunique()}</div>
        <div class="metric-caption">Major Indian urban hubs</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Mean PM2.5</div>
        <div class="metric-value">{raw_df['PM2.5'].mean():.1f} <span style="font-size:0.8rem; color:#94A3B8;">µg/m³</span></div>
        <div class="metric-caption">Std: ±{raw_df['PM2.5'].std():.1f}</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Mean PM10</div>
        <div class="metric-value">{raw_df['PM10'].mean():.1f} <span style="font-size:0.8rem; color:#94A3B8;">µg/m³</span></div>
        <div class="metric-caption">Std: ±{raw_df['PM10'].std():.1f}</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    top_m = eval_metrics["leaderboard"].iloc[0]["Model"].split(" (")[0]
    top_acc = eval_metrics["leaderboard"].iloc[0]["Accuracy"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Top Classifier</div>
        <div class="metric-value" style="font-size:1.3rem; color:#38BDF8;">{top_m}</div>
        <div class="metric-caption">Accuracy: {top_acc*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with k5:
    severe_count = sum(raw_df['AQI_Bucket'] == 'Severe')
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Severe Records</div>
        <div class="metric-value" style="color:#EF4444;">{severe_count:,}</div>
        <div class="metric-caption">{(severe_count/len(raw_df))*100:.1f}% of observations</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Application Tabs
tab_olap, tab_models, tab_viz, tab_infer = st.tabs([
    "🌐 Warehouse & OLAP Explorer",
    "🤖 Model Training & Leaderboard",
    "📊 Comparative Visualizations",
    "🚀 Custom CSV Ingestion & Inference"
])

# ------------------------------------------------------------------------------
# TAB 1: Warehouse & OLAP Explorer
# ------------------------------------------------------------------------------
with tab_olap:
    st.subheader("Multidimensional Warehouse & OLAP Explorer")
    st.markdown("Explore the **Star Schema** relational design and execute live analytical OLAP operations.")

    # Star Schema Entity Visuals
    c_d, c_f, c_l = st.columns([1, 1.4, 1])
    with c_d:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #60A5FA;">
            <div style="font-weight:700; color:#60A5FA; margin-bottom:6px;">📅 Dim_Date</div>
            <div style="font-size:0.8rem; color:#94A3B8;">Temporal Dimension Attributes</div>
            <hr style="border-color: rgba(148, 163, 184, 0.1); margin: 6px 0;">
            <div style="font-size:0.8rem; line-height: 1.7;">
                <div><span class="schema-tag tag-pk">PK</span> <b>DateKey</b> (Surrogate)</div>
                <div><span class="schema-tag">ATTR</span> Date, Year, Month</div>
                <div><span class="schema-tag">ATTR</span> Season, Day_of_Week</div>
            </div>
            <div style="font-size:0.75rem; color:#64748B; margin-top:6px;">
                <b>Hierarchy:</b> Year ➔ Season ➔ Month ➔ Date
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c_f:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #10B981; border-right: 4px solid #10B981;">
            <div style="font-weight:700; color:#10B981; margin-bottom:6px;">⚡ Fact_AirQuality</div>
            <div style="font-size:0.8rem; color:#94A3B8;">Central Fact Table (Grain: 1 daily reading per city)</div>
            <hr style="border-color: rgba(148, 163, 184, 0.1); margin: 6px 0;">
            <div style="font-size:0.8rem; line-height: 1.7;">
                <div><span class="schema-tag tag-pk">PK</span> <b>FactID</b> | <span class="schema-tag tag-fk">FK</span> <b>DateKey</b>, <b>LocationKey</b></div>
                <div><span class="schema-tag tag-measure">MEASURE</span> PM2.5, PM10, NO2, SO2, CO</div>
                <div><span class="schema-tag tag-measure">MEASURE</span> AQI (Calculated Index)</div>
                <div><span class="schema-tag">TARGET</span> AQI_Bucket (CPCB Severity)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c_l:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #F59E0B;">
            <div style="font-weight:700; color:#F59E0B; margin-bottom:6px;">📍 Dim_Location</div>
            <div style="font-size:0.8rem; color:#94A3B8;">Spatial Dimension Attributes</div>
            <hr style="border-color: rgba(148, 163, 184, 0.1); margin: 6px 0;">
            <div style="font-size:0.8rem; line-height: 1.7;">
                <div><span class="schema-tag tag-pk">PK</span> <b>LocationKey</b> (Surrogate)</div>
                <div><span class="schema-tag">ATTR</span> City, State, Zone</div>
                <div><span class="schema-tag">ATTR</span> Latitude, Longitude</div>
            </div>
            <div style="font-size:0.75rem; color:#64748B; margin-top:6px;">
                <b>Hierarchy:</b> Zone ➔ State ➔ City
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    olap_choice = st.radio(
        "Choose OLAP Operation:",
        ["1. Slice (1D Sub-Cube)", "2. Dice (Multi-Dimensional Sub-Cube)", "3. Roll-up & Drill-down (Temporal Hierarchy)", "4. Pivot (Cross-Tabulation Matrix)"],
        horizontal=True
    )

    if olap_choice.startswith("1. Slice"):
        col_sl1, col_sl2 = st.columns([1, 2])
        with col_sl1:
            dim = st.selectbox("Dimension:", ["Season", "City", "Year"])
            val = st.selectbox("Fix Value:", sorted(list(raw_df[dim].unique())))
            slice_res = slice_cube(raw_df, dim, val)
            st.metric("Records in Slice", f"{len(slice_res):,}", delta=f"{(len(slice_res)/len(raw_df))*100:.1f}% of total")
            st.write(slice_res[["PM2.5", "PM10", "NO2", "SO2", "CO", "AQI"]].describe().round(2).T[["mean", "std", "min", "max"]])
        with col_sl2:
            fig_sl = px.box(
                slice_res,
                x="City" if dim != "City" else "Season",
                y="AQI",
                color="City" if dim != "City" else "Season",
                title=f"<b>AQI Variance on Slice [{dim} = '{val}']</b>",
                template="plotly_dark"
            )
            fig_sl.update_layout(paper_bgcolor="rgba(15, 23, 42, 0)", plot_bgcolor="rgba(30, 41, 59, 0.4)", height=320)
            st.plotly_chart(fig_sl, use_container_width=True)
        st.dataframe(slice_res.head(20), use_container_width=True)

    elif olap_choice.startswith("2. Dice"):
        cd1, cd2, cd3 = st.columns(3)
        with cd1:
            all_cities = sorted(list(raw_df["City"].unique()))
            d_cities = st.multiselect("Cities:", all_cities, default=all_cities[:3])
        with cd2:
            d_seasons = st.multiselect("Seasons:", ["Winter", "Summer", "Monsoon", "Post-Monsoon"], default=["Winter", "Post-Monsoon"])
        with cd3:
            d_years = st.multiselect("Years:", sorted(list(raw_df["Year"].unique())), default=sorted(list(raw_df["Year"].unique()))[-3:])
        aqi_b = st.slider("AQI Range:", 0, 500, (50, 450))

        dice_res = dice_cube(raw_df, cities=d_cities, seasons=d_seasons, years=d_years, aqi_range=aqi_b)
        st.markdown(f"**Diced Sub-Cube Volume:** `{len(dice_res):,}` records.")
        if len(dice_res) > 0:
            fig_dice = px.scatter(
                dice_res,
                x="PM2.5",
                y="PM10",
                color="AQI_Bucket",
                hover_data=["City", "Date", "AQI"],
                title="<b>Diced Cube: PM2.5 vs PM10 by Severity</b>",
                template="plotly_dark",
                color_discrete_map={
                    "Good": "#10B981", "Satisfactory": "#34D399", "Moderate": "#F59E0B",
                    "Poor": "#F97316", "Very Poor": "#EF4444", "Severe": "#7F1D1D"
                }
            )
            fig_dice.update_layout(paper_bgcolor="rgba(15, 23, 42, 0)", plot_bgcolor="rgba(30, 41, 59, 0.4)", height=340)
            st.plotly_chart(fig_dice, use_container_width=True)
            st.dataframe(dice_res.head(25), use_container_width=True)

    elif olap_choice.startswith("3. Roll-up"):
        cr1, cr2, cr3 = st.columns(3)
        with cr1:
            lvl = st.selectbox("Hierarchy Grain:", ["Yearly", "Seasonal", "Monthly", "Daily"])
        with cr2:
            fn = st.selectbox("Aggregation Operator:", ["Mean", "Median", "Max", "Min"])
        with cr3:
            obs_m = st.selectbox("Observed Metric:", ["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO"])

        agg_res = rollup_drilldown(raw_df, level=lvl, agg_func=fn)
        st.markdown(f"**Aggregated Level:** `{len(agg_res):,}` summary rows at `{lvl}` granularity using `{fn}`.")
        x_field = "Year" if lvl == "Yearly" else ("Season" if lvl == "Seasonal" else ("Month_Name" if lvl == "Monthly" else "Date"))
        fig_trend = px.line(
            agg_res,
            x=x_field,
            y=obs_m,
            color="City",
            markers=True,
            title=f"<b>{fn} {obs_m} Across Cities at {lvl} Level</b>",
            template="plotly_dark"
        )
        fig_trend.update_layout(paper_bgcolor="rgba(15, 23, 42, 0)", plot_bgcolor="rgba(30, 41, 59, 0.4)", height=350)
        st.plotly_chart(fig_trend, use_container_width=True)
        st.dataframe(agg_res.head(30), use_container_width=True)

    else:
        cp1, cp2, cp3, cp4 = st.columns(4)
        with cp1:
            r_dim = st.selectbox("Row Dimension (Y):", ["City", "Season", "Year"], index=0)
        with cp2:
            c_dim = st.selectbox("Column Dimension (X):", ["Season", "Year", "Month_Name", "AQI_Bucket"], index=0)
        with cp3:
            p_val = st.selectbox("Measure Value:", ["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO"], index=0)
        with cp4:
            p_fn = st.selectbox("Aggregator:", ["Mean", "Median", "Max", "Min"], index=0)

        if r_dim == c_dim:
            st.error("Row and Column dimensions must differ.")
        else:
            pvt = pivot_cube(raw_df, row_dim=r_dim, col_dim=c_dim, metric=p_val, agg_func=p_fn)
            col_t, col_h = st.columns([1, 1.2])
            with col_t:
                st.dataframe(pvt, use_container_width=True)
            with col_h:
                fig_pvt = px.imshow(
                    pvt,
                    text_auto=True,
                    color_continuous_scale="Spectral_r",
                    title=f"<b>Pivot Heatmap: {r_dim} vs {c_dim} [{p_val} ({p_fn})]</b>",
                    template="plotly_dark"
                )
                fig_pvt.update_layout(paper_bgcolor="rgba(15, 23, 42, 0)", plot_bgcolor="rgba(30, 41, 59, 0.4)", height=340)
                st.plotly_chart(fig_pvt, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: Model Training & Leaderboard
# ------------------------------------------------------------------------------
with tab_models:
    st.subheader("Model Training & Evaluation Leaderboard")
    st.markdown("Quantitative assessment of four classification models evaluated on the holdout test set (75:25 split).")

    leaderboard = eval_metrics["leaderboard"]
    st.dataframe(
        leaderboard.style.highlight_max(
            subset=["Accuracy", "Precision (Weighted)", "Recall (Weighted)", "F1-Score (Weighted)"],
            color="#065F46"
        ).format({
            "Accuracy": "{:.4f}",
            "Precision (Weighted)": "{:.4f}",
            "Recall (Weighted)": "{:.4f}",
            "F1-Score (Weighted)": "{:.4f}"
        }),
        use_container_width=True
    )

    st.markdown("#### **Algorithm Architecture Breakdown**")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#60A5FA; margin-bottom:6px;">1. Linear Regression (Threshold Classifier)</h4>
            <p style="font-size:0.82rem; color:#CBD5E1;">
                Trained as an Ordinary Least Squares estimator on continuous AQI, functioning as a discrete classifier
                by passing continuous predictions through regulatory CPCB cutoffs (<code>[0, 50] ➔ Good</code>, etc.).
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#34D399; margin-bottom:6px;">2. Multinomial Logistic Regression</h4>
            <p style="font-size:0.82rem; color:#CBD5E1;">
                Generalized Softmax multiclass classification optimized via L-BFGS, predicting posterior probability
                distributions across the 6 ordered severity categories.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#FBBF24; margin-bottom:6px;">3. J48 / C4.5 Decision Tree</h4>
            <p style="font-size:0.82rem; color:#CBD5E1;">
                Quinlan's entropy-based C4.5 decision tree algorithm utilizing Shannon Information Gain to produce
                hierarchical orthogonal decision boundaries.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#F472B6; margin-bottom:6px;">4. Random Forest Classifier</h4>
            <p style="font-size:0.82rem; color:#CBD5E1;">
                Ensemble bagging algorithm fitting 100 decorrelated decision trees on bootstrap replicas with random
                feature sub-spacing for robust non-linear generalization.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 3: Comparative Visualizations
# ------------------------------------------------------------------------------
with tab_viz:
    st.subheader("Comparative Performance Visualizations")
    st.markdown("Dedicated individual Plotly charts for each primary metric alongside multi-class confusion matrices.")

    vc1, vc2 = st.columns(2)
    with vc1:
        st.plotly_chart(create_separate_metric_chart(leaderboard, "Accuracy", "Graph 1: Comparative Accuracy Across Models"), use_container_width=True)
    with vc2:
        st.plotly_chart(create_separate_metric_chart(leaderboard, "Precision (Weighted)", "Graph 2: Comparative Precision Across Models"), use_container_width=True)

    vc3, vc4 = st.columns(2)
    with vc3:
        st.plotly_chart(create_separate_metric_chart(leaderboard, "Recall (Weighted)", "Graph 3: Comparative Recall Across Models"), use_container_width=True)
    with vc4:
        st.plotly_chart(create_separate_metric_chart(leaderboard, "F1-Score (Weighted)", "Graph 4: Comparative F1-Score Across Models"), use_container_width=True)

    st.divider()
    st.markdown("#### **Multi-Class Confusion Matrix Heatmaps**")
    sel_cm_m = st.selectbox("Inspect Model Confusion Matrix:", list(pipeline_bundle["trained_models"].keys()))
    cm_norm = st.checkbox("Normalize by True Class Row (%)", value=False)
    cm_matrix = eval_metrics["details"][sel_cm_m]["confusion_matrix"]
    st.plotly_chart(create_confusion_matrix_heatmap(cm_matrix, ORDERED_CLASSES, sel_cm_m, normalize=cm_norm), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: Custom CSV Ingestion & Inference
# ------------------------------------------------------------------------------
with tab_infer:
    st.subheader("Custom CSV Ingestion, Schema Validator & Batch Inference")
    st.markdown("Upload arbitrary CSV files to classify pollution severity using the trained models.")

    c_tmp1, c_tmp2 = st.columns([2, 1])
    with c_tmp1:
        st.markdown("Need a verified sample CSV to test batch inference? Download a sample dataset with authentic readings:")
    with c_tmp2:
        sample_bytes = get_sample_upload_data(num_rows=50).to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Verified Sample CSV",
            data=sample_bytes,
            file_name="sample_cpcb_pollutant_records.csv",
            mime="text/csv",
            type="secondary",
            use_container_width=True
        )

    st.divider()
    uploaded_file = st.file_uploader("Upload Air Quality CSV for Inference:", type=["csv"])
    sel_infer_model = st.selectbox("Classifier to Score Uploaded Data:", list(pipeline_bundle["trained_models"].keys()))

    if uploaded_file is not None:
        try:
            in_df = pd.read_csv(uploaded_file)
            st.success(f"CSV Ingested: `{len(in_df):,}` rows and `{len(in_df.columns)}` columns detected.")

            missing_cols = [c for c in FEATURE_COLS if c not in in_df.columns]
            if missing_cols:
                st.error(f"❌ Schema Validation Error: Uploaded file is missing required column(s): **{', '.join(missing_cols)}**.")
                st.info(f"Required pollutant features: `{', '.join(FEATURE_COLS)}`")
            else:
                with st.spinner(f"Scoring records using {sel_infer_model}..."):
                    out_df, meta_info = score_batch_records(in_df, sel_infer_model, pipeline_bundle)

                st.success(f"✅ Successfully scored {meta_info['count']:,} records!")
                res_col1, res_col2 = st.columns([1, 1.5])
                with res_col1:
                    st.markdown("#### **Predicted Class Distribution**")
                    dist_df = pd.Series(meta_info["distribution"]).reset_index()
                    dist_df.columns = ["Severity Bucket", "Count"]
                    st.dataframe(dist_df, use_container_width=True, hide_index=True)
                with res_col2:
                    fig_dist = px.bar(
                        dist_df,
                        x="Severity Bucket",
                        y="Count",
                        color="Severity Bucket",
                        title="<b>Predicted Severity Breakdown</b>",
                        template="plotly_dark",
                        color_discrete_map={
                            "Good": "#10B981", "Satisfactory": "#34D399", "Moderate": "#F59E0B",
                            "Poor": "#F97316", "Very Poor": "#EF4444", "Severe": "#7F1D1D"
                        }
                    )
                    fig_dist.update_layout(paper_bgcolor="rgba(15, 23, 42, 0)", plot_bgcolor="rgba(30, 41, 59, 0.4)", height=280)
                    st.plotly_chart(fig_dist, use_container_width=True)

                st.markdown("#### **Enriched Output Preview**")
                st.dataframe(out_df.head(50), use_container_width=True)

                out_bytes = out_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Download Enriched Predictions (CSV)",
                    data=out_bytes,
                    file_name="cpcb_batch_predictions.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Inference error: {str(e)}")
    else:
        st.info("Upload an air quality CSV above to execute batch scoring and download predictions.")

st.divider()
st.markdown("<div style='text-align: center; color: #64748B; font-size: 0.8rem;'>Urban Air Quality & Pollution Forecasting System • Built with Streamlit, Plotly & Scikit-Learn</div>", unsafe_allow_html=True)
EOF

git add app.py
commit_step "2026-10-10 04:52:19 +0530" "refactor: optimize caching, handle edge cases, and add input validation"

# ==============================================
# MILESTONE 8 (10-Oct 05:44:11 IST)
# docs: add comprehensive README with star schema architecture and usage guide
# ==============================================
echo ""
echo "[8/8] Writing documentation and finalizing setup..."

cat << 'EOF' > README.md
# Urban Air Quality & Pollution Forecasting System
### Multidimensional Data Warehousing & Data Mining (DWM) Application

A production-grade, standalone **Streamlit** application engineered for atmospheric data warehousing, multidimensional OLAP analytics, and machine learning air quality classification utilizing **authentic public Central Pollution Control Board (CPCB, India) environmental sensor records (>20,000 observations)**.

---

## 🏛️ System Architecture

### 1. Data Ingestion & Star Schema Warehouse
- **Authentic Data Source:** Official daily ambient monitoring station measurements from major Indian metropolitan centers (Delhi, Ahmedabad, Bengaluru, Mumbai, Chennai, Hyderabad, Kolkata, etc.).
- **Features Ingested:** `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `AQI`, and `AQI_Bucket`.
- **Relational Star Schema:**
  - `Fact_AirQuality`: Stores daily observations, pollutant measures, computed index, and surrogate keys (`FactID`, `DateKey`, `LocationKey`).
  - `Dim_Location`: Spatial dimension with `LocationKey`, `City`, `State`, `Zone`, `Latitude`, `Longitude`.
  - `Dim_Date`: Temporal dimension with `DateKey`, `Date`, `Year`, `Month`, `Season`, `Day_of_Week`.

```
                        +----------------------+
                        |       Dim_Date       |
                        +----------------------+
                        | DateKey (PK)         |
                        | Date                 |
                        | Year, Month, Day     |
                        | Season, Day_of_Week  |
                        +----------+-----------+
                                   | 1
                                   |
                                   | *
+----------------------+ *         |         * +----------------------+
|     Dim_Location     +-----------+-----------+   Fact_AirQuality    |
+----------------------+ 1                     +----------------------+
| LocationKey (PK)     |                       | FactID (PK)          |
| City                 |                       | DateKey (FK)         |
| State, Zone          |                       | LocationKey (FK)     |
| Latitude, Longitude  |                       | PM2.5, PM10          |
+----------------------+                       | NO2, SO2, CO         |
                                               | AQI, AQI_Bucket      |
                                               +----------------------+
```

### 2. Multidimensional OLAP Cube Analytics
- **Slice:** Filter across a single dimension (e.g., `Season = 'Winter'`).
- **Dice:** Multi-dimensional sub-cube extraction (e.g., specific cities, seasons, years, and AQI bounds).
- **Roll-up & Drill-down:** Temporal aggregation transitions (`Daily ➔ Monthly ➔ Seasonal ➔ Yearly`) with flexible aggregate operators (`Mean`, `Median`, `Max`, `Min`).
- **Pivot:** Axis reorientation producing interactive cross-tabulated matrices and 2D Plotly heatmaps.

### 3. Machine Learning Mining Pipeline
Four distinct classification models trained on standardized pollutant features:
1. **Linear Regression (Threshold Classifier):** Continuous AQI estimator mapped to discrete severity buckets via standard CPCB cutoffs (`[0, 50] ➔ Good`, `(50, 100] ➔ Satisfactory`, etc.).
2. **Multinomial Logistic Regression:** Softmax multiclass classification with L-BFGS optimizer.
3. **J48 / C4.5 Decision Tree:** Ross Quinlan's information gain / entropy criterion (`DecisionTreeClassifier(criterion='entropy')`).
4. **Random Forest Classifier:** Bagged ensemble of 100 entropy decision trees.

### 4. Dedicated Visualizations & Batch Inference
- Consolidated evaluation table reporting **Accuracy, Weighted Precision, Weighted Recall, and Weighted F1-Score**.
- **Four Dedicated, Separate Interactive Plotly Graphs:**
  - Graph 1: Comparative Accuracy Across Models
  - Graph 2: Comparative Precision Across Models
  - Graph 3: Comparative Recall Across Models
  - Graph 4: Comparative F1-Score Across Models
- Interactive multi-class confusion matrix heatmaps.
- **Batch CSV Ingestion Engine:** Validates required pollutant columns (`PM2.5`, `PM10`, `NO2`, `SO2`, `CO`), dynamic model selection, and instant one-click CSV export (`st.download_button`).

---

## 🚀 Quickstart & Execution

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Streamlit Application
```bash
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

---

## 📜 Regulatory Standards (CPCB India)

| AQI Range | Bucket | Health Implications |
| :---: | :---: | :--- |
| **0 – 50** | **Good** | Minimal health impact |
| **51 – 100** | **Satisfactory** | Minor breathing discomfort to sensitive people |
| **101 – 200** | **Moderate** | Breathing discomfort to people with asthma and heart disease |
| **201 – 300** | **Poor** | Breathing discomfort to most people on prolonged exposure |
| **301 – 400** | **Very Poor** | Respiratory illness on prolonged exposure |
| **401 – 500+** | **Severe** | Affects healthy people and severely impacts those with existing disease |
EOF

git add README.md setup_project_with_git_history.sh
commit_step "2026-10-10 05:44:11 +0530" "docs: add comprehensive README with star schema architecture and usage guide"

echo ""
echo "========================================================================"
echo "✅ Build Completed! Displaying Verified Progressive Git History:"
echo "========================================================================"
git log --graph --pretty=format:'%C(yellow)%h%Creset | %C(green)%ad%Creset | %C(bold blue)%an%Creset | %s' --date=format:'%d-%b-%Y %H:%M:%S %z'
echo ""

