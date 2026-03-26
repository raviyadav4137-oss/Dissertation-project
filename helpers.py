"""
helpers.py
----------
Shared utility functions used across the EMDS pipeline.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def compute_metrics(y_true, y_pred, y_proba):
    """Return a dict of standard classification metrics."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "roc_auc":   roc_auc_score(y_true, y_proba)
    }


def print_metrics(metrics, model_name="Model"):
    print(f"\n[{model_name}] Performance Metrics:")
    for k, v in metrics.items():
        print(f"  {k.capitalize():<12}: {v:.4f}")


def set_plot_style():
    """Apply a clean consistent style to all matplotlib figures."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "font.family":      "DejaVu Sans",
        "font.size":        11,
    })


def sample_balanced(X, y, n_per_class=500, random_state=42):
    """
    Return a balanced subsample with n_per_class samples from each class.
    Useful for SHAP global analysis.
    """
    rng = np.random.default_rng(random_state)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    sampled_0 = rng.choice(idx_0, size=min(n_per_class, len(idx_0)), replace=False)
    sampled_1 = rng.choice(idx_1, size=min(n_per_class, len(idx_1)), replace=False)
    idx = np.concatenate([sampled_0, sampled_1])
    rng.shuffle(idx)
    return X[idx], y[idx]


def jaccard_similarity(set_a, set_b):
    """Jaccard similarity between two sets."""
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
