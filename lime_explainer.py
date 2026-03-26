"""
lime_explainer.py
-----------------
Generates LIME explanations for the trained XGBoost classifier
and compares them against SHAP for the same samples.

Produces:
  - Instance-level LIME bar charts for N_LOCAL_SAMPLES malware samples
  - Side-by-side SHAP vs LIME comparison figures
  - Metrics: consistency (Jaccard), processing time, simplicity score

Outputs saved to: explainability/outputs/lime/
"""

import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import shap

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = os.path.join("models", "saved")
SHAP_DIR      = os.path.join("explainability", "outputs", "shap")
OUTPUT_DIR    = os.path.join("explainability", "outputs", "lime")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_LOCAL_SAMPLES    = 5
N_LIME_FEATURES    = 10    # top features to show
N_LIME_PERTURB     = 5000  # perturbation samples per explanation
N_STABILITY_RUNS   = 5     # re-runs for Jaccard consistency


def load_artefacts():
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    X_train       = load(os.path.join(PROCESSED_DIR, "X_train.pkl"))
    X_test        = load(os.path.join(PROCESSED_DIR, "X_test.pkl"))
    y_test        = load(os.path.join(PROCESSED_DIR, "y_test.pkl"))
    feature_names = load(os.path.join(PROCESSED_DIR, "feature_names.pkl"))
    model         = load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    return model, X_train, X_test, y_test, feature_names


def build_lime_explainer(X_train, feature_names):
    """Initialise LIME tabular explainer using training data distribution."""
    print("[INFO] Building LIME explainer...")
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["Benign", "Malicious"],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )
    return explainer


def explain_sample(lime_explainer, model, sample, label="Malicious"):
    """Run LIME on a single sample and return the explanation object."""
    explanation = lime_explainer.explain_instance(
        data_row=sample,
        predict_fn=model.predict_proba,
        num_features=N_LIME_FEATURES,
        num_samples=N_LIME_PERTURB
    )
    return explanation


def plot_lime_bar(explanation, sample_id, label="Malicious"):
    """Save a LIME feature importance bar chart."""
    features_weights = explanation.as_list(label=1)
    features_weights.sort(key=lambda x: abs(x[1]), reverse=True)

    feat_labels = [fw[0] for fw in features_weights]
    weights     = [fw[1] for fw in features_weights]
    colours     = ["#ef4444" if w > 0 else "#3b82f6" for w in weights]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(feat_labels[::-1], weights[::-1], color=colours[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME Weight")
    ax.set_title(f"LIME Local Explanation — Sample {sample_id} ({label})")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"lime_local_sample_{sample_id}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def local_explanations(lime_explainer, model, X_test, y_test):
    """
    Generate LIME explanations for N_LOCAL_SAMPLES malicious samples.
    Returns list of (explanation, elapsed_time) tuples.
    """
    print(f"\n[INFO] Generating LIME local explanations for {N_LOCAL_SAMPLES} samples...")
    malware_idx  = np.where(y_test == 1)[0][:N_LOCAL_SAMPLES]
    explanations = []
    times        = []

    for i, idx in enumerate(malware_idx):
        sample = X_test[idx]
        start  = time.time()
        exp    = explain_sample(lime_explainer, model, sample)
        elapsed = time.time() - start
        times.append(elapsed)

        path = plot_lime_bar(exp, sample_id=i + 1)
        print(f"  Sample {i+1}: {elapsed:.2f}s  ->  {path}")
        explanations.append((idx, exp))

    avg_time = np.mean(times)
    print(f"\n  Average LIME time per sample: {avg_time:.2f}s  ({avg_time*1000:.0f} ms)")
    return explanations, times


def measure_lime_consistency(lime_explainer, model, sample, n_runs=N_STABILITY_RUNS):
    """
    Re-run LIME on the same sample n_runs times.
    Compute mean Jaccard similarity of top-10 feature sets.
    """
    print("\n[INFO] Measuring LIME explanation consistency...")
    feature_sets = []
    for _ in range(n_runs):
        exp    = explain_sample(lime_explainer, model, sample)
        top_features = set([fw[0] for fw in exp.as_list(label=1)])
        feature_sets.append(top_features)

    jaccard_scores = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            inter = len(feature_sets[i] & feature_sets[j])
            union = len(feature_sets[i] | feature_sets[j])
            jaccard_scores.append(inter / union if union > 0 else 1.0)

    mean_jaccard = np.mean(jaccard_scores)
    print(f"  Mean Jaccard similarity: {mean_jaccard:.4f}  (1.0 = perfectly consistent)")
    return mean_jaccard


def plot_shap_vs_lime(shap_explanation_path, lime_explanation_path, sample_id):
    """
    Load pre-saved SHAP and LIME figures and combine them into a side-by-side panel.
    """
    shap_img = plt.imread(shap_explanation_path)
    lime_img = plt.imread(lime_explanation_path)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].imshow(shap_img)
    axes[0].axis("off")
    axes[0].set_title(f"SHAP — Sample {sample_id}", fontsize=13, fontweight="bold")

    axes[1].imshow(lime_img)
    axes[1].axis("off")
    axes[1].set_title(f"LIME — Sample {sample_id}", fontsize=13, fontweight="bold")

    plt.suptitle(f"XAI Comparison: SHAP vs LIME — Sample {sample_id}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"comparison_shap_vs_lime_sample_{sample_id}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  EMDS — LIME Explanation Generation")
    print("=" * 60)

    model, X_train, X_test, y_test, feature_names = load_artefacts()

    lime_explainer = build_lime_explainer(X_train, feature_names)

    explanations, times = local_explanations(lime_explainer, model, X_test, y_test)

    # Consistency on first malware sample
    first_malware_idx = np.where(y_test == 1)[0][0]
    jaccard = measure_lime_consistency(lime_explainer, model, X_test[first_malware_idx])

    # Side-by-side comparison for each sample
    print("\n[INFO] Generating SHAP vs LIME comparison panels...")
    for i in range(1, N_LOCAL_SAMPLES + 1):
        shap_path = os.path.join(SHAP_DIR, f"shap_local_sample_{i}.png")
        lime_path = os.path.join(OUTPUT_DIR, f"lime_local_sample_{i}.png")
        if os.path.exists(shap_path) and os.path.exists(lime_path):
            plot_shap_vs_lime(shap_path, lime_path, sample_id=i)
        else:
            print(f"  [WARN] Missing SHAP output for sample {i} — run shap_explainer.py first.")

    # Save metrics
    lime_metrics = {
        "avg_time_per_sample_s": float(np.mean(times)),
        "avg_time_ms": float(np.mean(times) * 1000),
        "jaccard_consistency": float(jaccard),
        "n_features_shown": N_LIME_FEATURES,
        "n_perturbation_samples": N_LIME_PERTURB
    }
    with open(os.path.join(OUTPUT_DIR, "lime_metrics.pkl"), "wb") as f:
        pickle.dump(lime_metrics, f)

    print("\n[DONE] LIME explanation generation complete.")
    print(f"       Outputs saved to: {OUTPUT_DIR}/")
