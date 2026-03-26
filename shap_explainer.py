"""
shap_explainer.py
-----------------
Generates SHAP explanations for the trained XGBoost classifier.

Produces:
  - Global: SHAP summary plot (beeswarm), bar plot
  - Local:  Force plots and waterfall charts for 5 sample predictions
  - Metrics: Computation time, explanation stability scores

Outputs saved to: explainability/outputs/shap/
"""

import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = os.path.join("models", "saved")
OUTPUT_DIR    = os.path.join("explainability", "outputs", "shap")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BACKGROUND_SAMPLES = 1000   # samples used for SHAP global analysis
N_LOCAL_SAMPLES      = 5      # individual malware samples to explain


def load_artefacts():
    """Load test set, trained model, and feature names."""
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    X_test        = load(os.path.join(PROCESSED_DIR, "X_test.pkl"))
    y_test        = load(os.path.join(PROCESSED_DIR, "y_test.pkl"))
    feature_names = load(os.path.join(PROCESSED_DIR, "feature_names.pkl"))
    model         = load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    return model, X_test, y_test, feature_names


def build_explainer(model, X_background):
    """Create a SHAP TreeExplainer using a background sample."""
    print("[INFO] Building SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(
        model,
        data=shap.sample(X_background, N_BACKGROUND_SAMPLES),
        feature_perturbation="interventional"
    )
    return explainer


def global_explanations(explainer, X_sample, feature_names):
    """
    Compute SHAP values for a representative sample and produce:
      1. Beeswarm summary plot
      2. Mean absolute SHAP bar plot
    """
    print(f"\n[INFO] Computing global SHAP values for {len(X_sample):,} samples...")
    start = time.time()
    shap_values = explainer.shap_values(X_sample)
    elapsed = time.time() - start
    print(f"  Computation time: {elapsed:.2f}s  ({elapsed/len(X_sample)*1000:.1f} ms/sample)")

    # 1. Beeswarm summary plot
    print("  Generating beeswarm summary plot...")
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False, max_display=20
    )
    plt.title("SHAP Global Feature Importance (Beeswarm)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Bar plot (mean |SHAP|)
    print("  Generating bar plot...")
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False, max_display=20
    )
    plt.title("SHAP Global Feature Importance (Mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("  Global plots saved.")
    return shap_values, elapsed


def local_explanations(explainer, X_samples, y_samples, feature_names):
    """
    Generate SHAP waterfall chart for each of N_LOCAL_SAMPLES malware samples.
    """
    print(f"\n[INFO] Generating local SHAP explanations for {N_LOCAL_SAMPLES} samples...")

    # Select malware-positive samples
    malware_idx = np.where(y_samples == 1)[0][:N_LOCAL_SAMPLES]

    for i, idx in enumerate(malware_idx):
        sample = X_samples[idx:idx+1]
        sv     = explainer(sample)

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(sv[0], max_display=15, show=False)
        plt.title(f"SHAP Local Explanation — Sample {i+1} (Malicious)")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"shap_local_sample_{i+1}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


def measure_stability(explainer, X_test, n_repeats=5, sample_idx=0):
    """
    Measure SHAP explanation stability by re-running on the same sample
    n_repeats times and computing std of top-5 SHAP values.
    Ideal: std ≈ 0 (SHAP TreeExplainer is deterministic).
    """
    print("\n[INFO] Measuring SHAP stability...")
    sample = X_test[sample_idx:sample_idx+1]
    runs   = []

    for _ in range(n_repeats):
        sv = explainer.shap_values(sample)
        runs.append(sv[0])

    runs_arr = np.array(runs)        # shape: (n_repeats, n_features)
    top5_idx = np.argsort(np.abs(runs_arr[0]))[-5:]
    stds     = runs_arr[:, top5_idx].std(axis=0)

    print(f"  Std of top-5 SHAP values across {n_repeats} runs: {stds}")
    print(f"  Mean std: {stds.mean():.6f}  (0.000000 = perfectly stable)")
    return stds


def measure_rank_consistency(shap_values, n_seeds=3, top_n=10):
    """
    Compare feature importance rankings across random subsamples.
    Uses Spearman rank correlation of mean |SHAP| rankings.
    """
    print("\n[INFO] Measuring SHAP rank consistency across subsamples...")
    rng     = np.random.default_rng(42)
    n       = shap_values.shape[0]
    rankings = []

    for seed in range(n_seeds):
        idx    = rng.choice(n, size=n // 3, replace=False)
        mean_abs = np.abs(shap_values[idx]).mean(axis=0)
        rank   = np.argsort(-mean_abs)[:top_n]
        rankings.append(rank)

    corrs = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            r, _ = spearmanr(rankings[i], rankings[j])
            corrs.append(r)

    mean_corr = np.mean(corrs)
    print(f"  Spearman rank correlation (avg): {mean_corr:.4f}  (1.0 = perfect consistency)")
    return mean_corr


if __name__ == "__main__":
    print("=" * 60)
    print("  EMDS — SHAP Explanation Generation")
    print("=" * 60)

    model, X_test, y_test, feature_names = load_artefacts()

    explainer = build_explainer(model, X_test)

    # Use a representative 1,000-sample subset for global plots
    sample_idx    = np.random.choice(len(X_test), N_BACKGROUND_SAMPLES, replace=False)
    X_global_sample = X_test[sample_idx]
    y_global_sample = y_test[sample_idx]

    shap_values, global_time = global_explanations(explainer, X_global_sample, feature_names)

    local_explanations(explainer, X_test, y_test, feature_names)

    stability = measure_stability(explainer, X_test)

    rank_corr = measure_rank_consistency(shap_values)

    # Save metrics
    shap_metrics = {
        "global_computation_time_s": global_time,
        "ms_per_sample": global_time / N_BACKGROUND_SAMPLES * 1000,
        "stability_std_mean": float(stability.mean()),
        "rank_consistency_spearman": float(rank_corr)
    }
    with open(os.path.join(OUTPUT_DIR, "shap_metrics.pkl"), "wb") as f:
        pickle.dump(shap_metrics, f)

    print("\n[DONE] SHAP explanation generation complete.")
    print(f"       Outputs saved to: {OUTPUT_DIR}/")
