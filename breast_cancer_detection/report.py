"""HTML report generation: matplotlib charts embedded as base64 images plus metrics tables."""
import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)


def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _plot_class_balance(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    counts = df["target"].map({0: "Benign", 1: "Malignant"}).value_counts()
    ax.bar(counts.index, counts.values, color=["#2a9d8f", "#e76f51"])
    ax.set_title("Class Balance")
    ax.set_ylabel("Sample Count")
    return _fig_to_base64(fig)


def _plot_confusion_matrix(y_test, y_pred) -> str:
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Benign", "Malignant"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_to_base64(fig)


def _plot_precision_recall(y_test, y_proba) -> str:
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(recall, precision, color="#e76f51", lw=2, label=f"AP = {ap_score:.3f}")
    ax.fill_between(recall, precision, alpha=0.15, color="#e76f51")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def _plot_roc_curve(y_test, y_proba) -> str:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, color="#264653", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def _plot_feature_importance(feature_importance: pd.Series, top_n: int = 10) -> str:
    top = feature_importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.barh(top.index, top.values, color="#2a9d8f")
    ax.set_title(f"Top {top_n} Feature Importances (Permutation)")
    ax.set_xlabel("Mean Importance Decrease")
    return _fig_to_base64(fig)


def _metrics_table_html(metrics: Dict[str, float]) -> str:
    tiles = "".join(
        f'<div class="tile"><div class="tile-value">{value:.3f}</div>'
        f'<div class="tile-label">{name}</div></div>'
        for name, value in metrics.items()
    )
    return f'<div class="tiles">{tiles}</div>'


def _correlations_table_html(top_correlations: List[dict]) -> str:
    rows = "".join(
        f'<tr><td>{c["feature_a"]}</td><td>{c["feature_b"]}</td><td>{c["correlation"]:.3f}</td></tr>'
        for c in top_correlations
    )
    return f"""
    <table>
        <thead><tr><th>Feature A</th><th>Feature B</th><th>|Correlation|</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """


_STYLE = """
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; background: #f7f7f5; color: #1f2937; margin: 0; padding: 2rem; }
.container { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
.meta { color: #6b7280; margin-bottom: 2rem; }
h2 { font-size: 1.2rem; margin-top: 2.5rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.4rem; }
.tiles { display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 1rem; }
.tile { background: white; border-radius: 10px; padding: 1rem 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); min-width: 120px; text-align: center; }
.tile-value { font-size: 1.6rem; font-weight: 700; color: #264653; }
.tile-label { color: #6b7280; font-size: 0.85rem; margin-top: 0.25rem; }
.charts { display: flex; flex-wrap: wrap; gap: 1.5rem; margin-top: 1rem; }
.charts img { background: white; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); padding: 0.5rem; max-width: 100%; }
table { border-collapse: collapse; width: 100%; margin-top: 1rem; background: white; border-radius: 10px; overflow: hidden; }
th, td { text-align: left; padding: 0.6rem 1rem; border-bottom: 1px solid #e5e7eb; }
th { background: #264653; color: white; }
.params { background: white; border-radius: 10px; padding: 1rem 1.5rem; margin-top: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
"""


def generate_report(
    df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metrics: Dict[str, float],
    feature_importance: pd.Series,
    data_summary: dict,
    best_params: dict,
    candidates_evaluated: int,
    output_path: Path,
) -> Path:
    class_balance_img = _plot_class_balance(df)
    cm_img = _plot_confusion_matrix(y_test, y_pred)
    pr_img = _plot_precision_recall(y_test, y_proba)
    roc_img = _plot_roc_curve(y_test, y_proba)
    importance_img = _plot_feature_importance(feature_importance)

    params_html = "".join(f"<div><strong>{k}</strong>: {v}</div>" for k, v in best_params.items())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Breast Cancer Detection Report</title>
<style>{_STYLE}</style>
</head>
<body>
<div class="container">
    <h1>Breast Cancer Detection Report</h1>
    <div class="meta">
        Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} &middot;
        {df.shape[0]} total samples &middot; {X_test.shape[0]} test samples
    </div>

    <h2>Model Performance</h2>
    {_metrics_table_html(metrics)}

    <h2>Dataset Overview</h2>
    <div class="charts"><img src="data:image/png;base64,{class_balance_img}" alt="Class balance"></div>
    <h3>Most Correlated Feature Pairs</h3>
    {_correlations_table_html(data_summary["top_correlations"])}

    <h2>Evaluation Charts</h2>
    <div class="charts">
        <img src="data:image/png;base64,{cm_img}" alt="Confusion Matrix">
        <img src="data:image/png;base64,{pr_img}" alt="Precision-Recall Curve">
        <img src="data:image/png;base64,{roc_img}" alt="ROC Curve">
    </div>

    <h2>Feature Importance</h2>
    <div class="charts"><img src="data:image/png;base64,{importance_img}" alt="Feature importance"></div>

    <h2>Best Hyperparameters</h2>
    <div class="params">
        {params_html}
        <div style="margin-top: 0.5rem; color: #6b7280;">{candidates_evaluated} candidates evaluated</div>
    </div>
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
