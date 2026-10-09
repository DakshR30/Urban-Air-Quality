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
