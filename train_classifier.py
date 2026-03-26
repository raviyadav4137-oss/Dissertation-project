"""
train_classifier.py
-------------------
Trains XGBoost (primary) and Random Forest (comparison) classifiers
on the prepared EMBER feature splits.

Outputs:
  models/saved/xgboost_model.pkl
  models/saved/random_forest_model.pkl
  models/saved/results_summary.pkl   <- metrics dict for dashboard
  models/saved/confusion_matrix_*.png
  models/saved/roc_curve_*.png
"""

import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = os.path.join("models", "saved")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    """Load processed splits from disk."""
    def load(name):
        with open(os.path.join(PROCESSED_DIR, f"{name}.pkl"), "rb") as f:
            return pickle.load(f)

    return (
        load("X_train"), load("X_val"), load("X_test"),
        load("y_train"), load("y_val"), load("y_test"),
        load("feature_names")
    )


# ── Hyperparameter grids ──────────────────────────────────────────────────────
XGB_PARAM_GRID = {
    "n_estimators":      [100, 300],
    "max_depth":         [4, 6],
    "learning_rate":     [0.05, 0.1],
    "subsample":         [0.8, 1.0],
    "colsample_bytree":  [0.8, 1.0],
}

RF_PARAM_GRID = {
    "n_estimators":    [100, 300],
    "max_depth":       [None, 10],
    "min_samples_split": [2, 5],
}


def tune_and_train(estimator, param_grid, X_train, y_train, model_name):
    """Run 5-fold stratified grid search and return the best estimator."""
    print(f"\n[INFO] Tuning {model_name}...")
    cv = StratifiedKFold(n_splits=5, shuffle=False)

    search = GridSearchCV(
        estimator, param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"  Best params : {search.best_params_}")
    print(f"  Best CV F1  : {search.best_score_:.4f}")
    print(f"  Tuning time : {elapsed:.1f}s")
    return search.best_estimator_


def evaluate(model, X_test, y_test, model_name):
    """Compute and print all evaluation metrics."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }

    print(f"\n[RESULTS] {model_name} — Test Set Performance:")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")

    return metrics, y_pred, y_proba


def plot_confusion_matrix(y_test, y_pred, model_name):
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Malicious"],
        yticklabels=["Benign", "Malicious"],
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = os.path.join(MODELS_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curve(y_test, y_proba, model_name):
    """Save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(MODELS_DIR, f"roc_curve_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def save_model(model, name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  EMDS — Classifier Training")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_data()
    print(f"[INFO] Training set: {X_train.shape[0]:,} samples")
    print(f"[INFO] Test set:     {X_test.shape[0]:,} samples")

    # ── Combine train + val for final model fit ───────────────────────────────
    X_fit = np.concatenate([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])

    results_summary = {}

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_base = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb_model = tune_and_train(xgb_base, XGB_PARAM_GRID, X_train, y_train, "XGBoost")
    xgb_model.fit(X_fit, y_fit)

    xgb_metrics, xgb_pred, xgb_proba = evaluate(xgb_model, X_test, y_test, "XGBoost")
    plot_confusion_matrix(y_test, xgb_pred, "XGBoost")
    plot_roc_curve(y_test, xgb_proba, "XGBoost")
    save_model(xgb_model, "xgboost_model")
    results_summary["xgboost"] = xgb_metrics

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_model = tune_and_train(rf_base, RF_PARAM_GRID, X_train, y_train, "Random Forest")
    rf_model.fit(X_fit, y_fit)

    rf_metrics, rf_pred, rf_proba = evaluate(rf_model, X_test, y_test, "Random Forest")
    plot_confusion_matrix(y_test, rf_pred, "Random Forest")
    plot_roc_curve(y_test, rf_proba, "Random Forest")
    save_model(rf_model, "random_forest_model")
    results_summary["random_forest"] = rf_metrics

    # ── Save combined results ─────────────────────────────────────────────────
    results_path = os.path.join(MODELS_DIR, "results_summary.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results_summary, f)
    print(f"\n[INFO] Results summary saved: {results_path}")

    print("\n[DONE] Training complete.")
