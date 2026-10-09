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
