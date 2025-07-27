import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# ---------------------------
# Utility Functions
# ---------------------------
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ---------------------------
# Model Loading
# ---------------------------
def load_model(model_path: str):
    """Load a trained model from disk."""
    print(f"Loading model from {model_path}")
    return joblib.load(model_path)

# ---------------------------
# Evaluation
# ---------------------------
def evaluate_classification_model(model, X_test, y_test):
    """Compute performance metrics and return predictions."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }

    return y_pred, y_proba, metrics

def save_metrics(metrics: dict, output_path: str):
    """Save evaluation metrics to a CSV file."""
    df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    ensure_dir(output_path)
    df.to_csv(output_path, index=False)
    print(f"Saved metrics to {output_path}")

# ---------------------------
# Visualization
# ---------------------------
def plot_confusion_matrix(y_test, y_pred, save_path=None, normalize=False):
    """Plot and optionally save the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred, normalize="true" if normalize else None)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved confusion matrix to {save_path}")
    plt.show()

def plot_roc_pr_curves(y_test, y_proba, roc_path=None, pr_path=None):
    """Plot and save ROC and Precision-Recall curves."""
    if y_proba is None:
        print("Model does not support probability predictions. Skipping ROC/PR plots.")
        return

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC Curve", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    if roc_path:
        ensure_dir(roc_path)
        plt.savefig(roc_path, bbox_inches="tight", dpi=300)
        print(f"Saved ROC curve to {roc_path}")
    plt.show()

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="PR Curve", color="green")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    if pr_path:
        ensure_dir(pr_path)
        plt.savefig(pr_path, bbox_inches="tight", dpi=300)
        print(f"Saved PR curve to {pr_path}")
    plt.show()
