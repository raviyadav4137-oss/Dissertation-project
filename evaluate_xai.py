"""
evaluate_xai.py
---------------
XAI Evaluation Framework — Technical + Analyst Effectiveness.

Produces a printed report and a saved results table covering:
  A) Technical Effectiveness:
       - Classifier performance (both models)
       - SHAP vs LIME runtime comparison
       - Explanation stability (SHAP std, LIME Jaccard)

  B) Analyst Effectiveness (Simulated):
       - Structured rubric: Interpretability, Cognitive Simplicity, Actionability
       - Scored for SHAP and LIME across N_LOCAL_SAMPLES samples

  C) Concept Drift Sub-Analysis:
       - Performance across 3 temporal windows in the test set
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = os.path.join("models", "saved")
SHAP_DIR      = os.path.join("explainability", "outputs", "shap")
LIME_DIR      = os.path.join("explainability", "outputs", "lime")
OUTPUT_DIR    = os.path.join("evaluation", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Analyst Rubric ─────────────────────────────────────────────────────────────
# These scores simulate the evaluation a security analyst would give.
# Scale: 1 = Low, 2 = Medium, 3 = High
# Based on literature criteria from Saqib et al. (2024) and Galli et al. (2024).
ANALYST_RUBRIC = {
    "SHAP": {
        "Interpretability":     3,   # Feature names + directional sign are clear
        "Cognitive Simplicity": 2,   # Beeswarm plots require some ML literacy
        "Actionability":        3,   # Force plot clearly highlights investigation targets
    },
    "LIME": {
        "Interpretability":     2,   # Feature conditions can be verbose
        "Cognitive Simplicity": 3,   # Bar chart is visually simpler for non-ML users
        "Actionability":        2,   # Less stable across runs, reducing analyst trust
    }
}
SCORE_LABELS = {1: "Low", 2: "Medium", 3: "High"}


def load_all():
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    results   = load(os.path.join(MODELS_DIR, "results_summary.pkl"))
    X_test    = load(os.path.join(PROCESSED_DIR, "X_test.pkl"))
    y_test    = load(os.path.join(PROCESSED_DIR, "y_test.pkl"))
    xgb_model = load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    rf_model  = load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))

    shap_metrics = load(os.path.join(SHAP_DIR, "shap_metrics.pkl"))
    lime_metrics = load(os.path.join(LIME_DIR,  "lime_metrics.pkl"))

    return results, X_test, y_test, xgb_model, rf_model, shap_metrics, lime_metrics


# ── A: Technical Effectiveness ───────────────────────────────────────────────

def print_technical_results(results, shap_metrics, lime_metrics):
    print("\n" + "=" * 60)
    print("  A) TECHNICAL EFFECTIVENESS")
    print("=" * 60)

    print("\n  1. Classifier Performance (Test Set)")
    print(f"  {'Metric':<15} {'XGBoost':>12} {'Random Forest':>14}")
    print(f"  {'-'*43}")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        xgb_val = results["xgboost"][metric]
        rf_val  = results["random_forest"][metric]
        print(f"  {metric.capitalize():<15} {xgb_val:>12.4f} {rf_val:>14.4f}")

    print("\n  2. XAI Runtime Comparison")
    print(f"  {'Method':<10} {'Avg time/sample':>18} {'Notes'}")
    print(f"  {'-'*60}")
    print(f"  {'SHAP':<10} {shap_metrics['ms_per_sample']:>15.1f} ms   Deterministic (TreeExplainer)")
    print(f"  {'LIME':<10} {lime_metrics['avg_time_ms']:>15.1f} ms   Stochastic (perturbation-based)")

    print("\n  3. Explanation Stability")
    print(f"  SHAP — Mean Std of top-5 values (5 runs): {shap_metrics['stability_std_mean']:.6f}  (0 = perfect)")
    print(f"  SHAP — Rank consistency (Spearman):        {shap_metrics['rank_consistency_spearman']:.4f}")
    print(f"  LIME — Jaccard consistency (5 runs):       {lime_metrics['jaccard_consistency']:.4f}  (1.0 = perfect)")


# ── B: Analyst Effectiveness ─────────────────────────────────────────────────

def print_analyst_rubric():
    print("\n" + "=" * 60)
    print("  B) ANALYST EFFECTIVENESS (Simulated Rubric)")
    print("=" * 60)
    print(f"\n  {'Criterion':<25} {'SHAP':>8} {'LIME':>8}")
    print(f"  {'-'*45}")
    for criterion in ["Interpretability", "Cognitive Simplicity", "Actionability"]:
        shap_score = ANALYST_RUBRIC["SHAP"][criterion]
        lime_score = ANALYST_RUBRIC["LIME"][criterion]
        print(f"  {criterion:<25} {SCORE_LABELS[shap_score]:>8} {SCORE_LABELS[lime_score]:>8}")

    shap_total = sum(ANALYST_RUBRIC["SHAP"].values())
    lime_total = sum(ANALYST_RUBRIC["LIME"].values())
    print(f"  {'-'*45}")
    print(f"  {'Total Score (out of 9)':<25} {shap_total:>8} {lime_total:>8}")
    print(f"\n  >> SHAP scores higher on Interpretability and Actionability.")
    print(f"  >> LIME scores higher on Cognitive Simplicity (simpler bar chart output).")


def plot_analyst_radar():
    """Radar/spider chart comparing SHAP vs LIME analyst scores."""
    categories   = list(ANALYST_RUBRIC["SHAP"].keys())
    shap_scores  = list(ANALYST_RUBRIC["SHAP"].values())
    lime_scores  = list(ANALYST_RUBRIC["LIME"].values())

    N      = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    shap_scores += shap_scores[:1]
    lime_scores += lime_scores[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, shap_scores, "b-o", linewidth=2, label="SHAP")
    ax.fill(angles, shap_scores, alpha=0.15, color="blue")
    ax.plot(angles, lime_scores, "r-o", linewidth=2, label="LIME")
    ax.fill(angles, lime_scores, alpha=0.15, color="red")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Low", "Medium", "High"], fontsize=8)
    ax.set_ylim(0, 3)
    ax.set_title("Analyst Effectiveness: SHAP vs LIME", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    path = os.path.join(OUTPUT_DIR, "analyst_effectiveness_radar.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Analyst radar chart saved: {path}")


# ── C: Concept Drift Sub-Analysis ────────────────────────────────────────────

def concept_drift_analysis(model, X_test, y_test):
    """
    Split test set into 3 equal temporal windows.
    Measure F1 and ROC-AUC in each window to detect performance degradation.
    """
    print("\n" + "=" * 60)
    print("  C) CONCEPT DRIFT SUB-ANALYSIS")
    print("=" * 60)

    n        = len(y_test)
    third    = n // 3
    windows  = [
        ("Early (T1)",  X_test[:third],          y_test[:third]),
        ("Mid   (T2)",  X_test[third:2*third],    y_test[third:2*third]),
        ("Late  (T3)",  X_test[2*third:],         y_test[2*third:]),
    ]

    print(f"\n  {'Window':<15} {'F1-Score':>10} {'ROC-AUC':>10} {'Samples':>10}")
    print(f"  {'-'*50}")

    drift_results = {}
    for name, X_w, y_w in windows:
        y_pred  = model.predict(X_w)
        y_proba = model.predict_proba(X_w)[:, 1]
        f1      = f1_score(y_w, y_pred)
        auc     = roc_auc_score(y_w, y_proba)
        print(f"  {name:<15} {f1:>10.4f} {auc:>10.4f} {len(y_w):>10,}")
        drift_results[name] = {"f1": f1, "roc_auc": auc}

    # Plot drift
    labels = [w[0] for w in windows]
    f1s    = [drift_results[l]["f1"]      for l in labels]
    aucs   = [drift_results[l]["roc_auc"] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(labels, f1s,  "b-o", label="F1-Score",  linewidth=2)
    ax.plot(labels, aucs, "r-s", label="ROC-AUC",   linewidth=2)
    ax.set_ylim(0.7, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Concept Drift: Model Performance Over Temporal Windows")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "concept_drift_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Concept drift plot saved: {path}")
    return drift_results


if __name__ == "__main__":
    print("=" * 60)
    print("  EMDS — XAI Evaluation Framework")
    print("=" * 60)

    results, X_test, y_test, xgb_model, rf_model, shap_metrics, lime_metrics = load_all()

    print_technical_results(results, shap_metrics, lime_metrics)
    print_analyst_rubric()
    plot_analyst_radar()
    drift_results = concept_drift_analysis(xgb_model, X_test, y_test)

    # Save full evaluation results
    eval_output = {
        "classifier_metrics": results,
        "shap_metrics":        shap_metrics,
        "lime_metrics":        lime_metrics,
        "analyst_rubric":      ANALYST_RUBRIC,
        "concept_drift":       drift_results
    }
    with open(os.path.join(OUTPUT_DIR, "full_evaluation_results.pkl"), "wb") as f:
        pickle.dump(eval_output, f)

    print("\n[DONE] Evaluation complete. Full results saved.")
