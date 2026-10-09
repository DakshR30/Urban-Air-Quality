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
