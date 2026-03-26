"""
app.py
------
EMDS Streamlit Analyst Dashboard

Run with:  streamlit run dashboard/app.py

Panels:
  1. Sidebar    — model selector, sample selector
  2. Header     — risk score gauge + prediction confidence
  3. SHAP panel — global summary + local force plot
  4. LIME panel — instance bar chart
  5. Comparison — side-by-side SHAP vs LIME
  6. Model KPIs — accuracy, precision, recall, F1, ROC-AUC
"""

import os
import sys
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Lazy imports (only when artefacts exist) ──────────────────────────────────
@st.cache_resource
def load_artefacts():
    """Load all saved models and processed data. Cached across sessions."""
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    try:
        xgb_model     = load(os.path.join("models", "saved", "xgboost_model.pkl"))
        rf_model      = load(os.path.join("models", "saved", "random_forest_model.pkl"))
        results       = load(os.path.join("models", "saved", "results_summary.pkl"))
        X_test        = load(os.path.join("data", "processed", "X_test.pkl"))
        y_test        = load(os.path.join("data", "processed", "y_test.pkl"))
        feature_names = load(os.path.join("data", "processed", "feature_names.pkl"))
        return xgb_model, rf_model, results, X_test, y_test, feature_names, None
    except FileNotFoundError as e:
        return None, None, None, None, None, None, str(e)


def get_shap_explanation(model, X_sample, feature_names):
    import shap
    explainer  = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return explainer, shap_values


def get_lime_explanation(model, X_train_sample, X_sample, feature_names):
    from lime.lime_tabular import LimeTabularExplainer
    explainer = LimeTabularExplainer(
        training_data=X_train_sample,
        feature_names=feature_names,
        class_names=["Benign", "Malicious"],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )
    exp = explainer.explain_instance(
        data_row=X_sample[0],
        predict_fn=model.predict_proba,
        num_features=10,
        num_samples=500   # reduced for dashboard responsiveness
    )
    return exp


def plot_shap_local(shap_values, sample_idx=0):
    import shap
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(shap_values[sample_idx], max_display=12, show=False)
    plt.title("SHAP Local Explanation — Feature Contributions")
    plt.tight_layout()
    return fig


def plot_lime_bar(lime_exp):
    features_weights = lime_exp.as_list(label=1)
    features_weights.sort(key=lambda x: abs(x[1]), reverse=True)
    labels  = [fw[0] for fw in features_weights]
    weights = [fw[1] for fw in features_weights]
    colours = ["#ef4444" if w > 0 else "#3b82f6" for w in weights]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(labels[::-1], weights[::-1], color=colours[::-1], edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME Weight (red = pushes toward Malicious, blue = toward Benign)")
    ax.set_title("LIME Local Explanation — Feature Contributions")
    plt.tight_layout()
    return fig


def plot_shap_global(model, X_sample, feature_names):
    import shap
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    fig, ax     = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, max_display=15)
    plt.title("SHAP Global Feature Importance (Beeswarm)")
    plt.tight_layout()
    return fig


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EMDS — Explainable Malware Detection",
    page_icon="🛡️",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛡️ Explainable Malware Detection System (EMDS)")
st.markdown(
    "**Dissertation Project KAS_004** — Interactive SOC Analyst Dashboard  \n"
    "Classifies Windows PE samples using XGBoost and explains predictions via SHAP and LIME."
)
st.divider()

# ── Load artefacts ────────────────────────────────────────────────────────────
xgb_model, rf_model, results, X_test, y_test, feature_names, load_error = load_artefacts()

if load_error:
    st.error(f"⚠️ Could not load model artefacts. Have you run all pipeline scripts?\n\n`{load_error}`")
    st.info(
        "**Setup steps:**\n"
        "1. `python data/prepare_dataset.py`\n"
        "2. `python models/train_classifier.py`\n"
        "3. `python explainability/shap_explainer.py`\n"
        "4. `python explainability/lime_explainer.py`\n"
        "5. `streamlit run dashboard/app.py`"
    )
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    model_choice = st.selectbox("Classifier", ["XGBoost (Primary)", "Random Forest"])
    model        = xgb_model if "XGBoost" in model_choice else rf_model

    st.subheader("Sample Selection")
    sample_mode = st.radio("Select sample by:", ["Index", "Malware samples only"])

    if sample_mode == "Index":
        sample_idx = st.slider("Sample index", 0, len(X_test) - 1, 0)
    else:
        malware_indices = np.where(y_test == 1)[0]
        rank = st.slider("Malware sample rank", 1, min(50, len(malware_indices)), 1)
        sample_idx = int(malware_indices[rank - 1])

    st.subheader("Explanation Settings")
    show_global  = st.checkbox("Show global SHAP summary", value=True)
    show_compare = st.checkbox("Show SHAP vs LIME comparison", value=True)
    st.divider()
    st.caption("EMDS — KAS_004 Dissertation  \nStreamlit Prototype v1.0")

# ── Selected Sample ───────────────────────────────────────────────────────────
X_sample = X_test[sample_idx:sample_idx + 1]
true_label = "🔴 Malicious" if y_test[sample_idx] == 1 else "🟢 Benign"

# ── Prediction ────────────────────────────────────────────────────────────────
pred_proba   = model.predict_proba(X_sample)[0]
pred_label   = "🔴 Malicious" if pred_proba[1] >= 0.5 else "🟢 Benign"
confidence   = max(pred_proba)
risk_score   = int(pred_proba[1] * 100)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Prediction",   pred_label)
col2.metric("True Label",   true_label)
col3.metric("Risk Score",   f"{risk_score} / 100")
col4.metric("Confidence",   f"{confidence*100:.1f}%")

st.progress(risk_score, text=f"Maliciousness Risk: {risk_score}%")
st.divider()

# ── SHAP Explanation ──────────────────────────────────────────────────────────
st.subheader("🔍 SHAP Explanation")
with st.spinner("Computing SHAP values..."):
    start = time.time()
    _, shap_values = get_shap_explanation(model, X_sample, feature_names)
    shap_time = (time.time() - start) * 1000

st.caption(f"⏱️ SHAP computation time: {shap_time:.1f} ms")
fig_shap = plot_shap_local(shap_values)
st.pyplot(fig_shap)
plt.close()

# ── LIME Explanation ──────────────────────────────────────────────────────────
st.subheader("🔎 LIME Explanation")
with st.spinner("Computing LIME explanation (this takes a few seconds)..."):
    # Use a background sample from test set for LIME training distribution
    bg_size   = min(200, len(X_test))
    bg_sample = X_test[:bg_size]
    start     = time.time()
    lime_exp  = get_lime_explanation(model, bg_sample, X_sample, feature_names)
    lime_time = (time.time() - start) * 1000

st.caption(f"⏱️ LIME computation time: {lime_time:.1f} ms")
fig_lime = plot_lime_bar(lime_exp)
st.pyplot(fig_lime)
plt.close()

# ── SHAP vs LIME Comparison ───────────────────────────────────────────────────
if show_compare:
    st.divider()
    st.subheader("⚖️ SHAP vs LIME Comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**SHAP — Shapley Values**")
        st.pyplot(plot_shap_local(shap_values))
        plt.close()
    with c2:
        st.markdown("**LIME — Perturbation Surrogate**")
        st.pyplot(plot_lime_bar(lime_exp))
        plt.close()

    st.caption(
        "SHAP provides a theoretically grounded attribution using game theory. "
        "LIME uses local perturbations to approximate model behaviour. "
        "Comparing both helps assess explanation consistency."
    )

# ── Global SHAP Summary ───────────────────────────────────────────────────────
if show_global:
    st.divider()
    st.subheader("🌐 Global SHAP Feature Importance")
    with st.spinner("Computing global SHAP summary (1,000 samples)..."):
        n_global = min(1000, len(X_test))
        X_global = X_test[:n_global]
        fig_global = plot_shap_global(model, X_global, feature_names)
    st.pyplot(fig_global)
    plt.close()

# ── Model Performance KPIs ────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Model Performance (Test Set)")

model_key = "xgboost" if "XGBoost" in model_choice else "random_forest"
m         = results[model_key]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Accuracy",  f"{m['accuracy']:.4f}")
k2.metric("Precision", f"{m['precision']:.4f}")
k3.metric("Recall",    f"{m['recall']:.4f}")
k4.metric("F1-Score",  f"{m['f1']:.4f}")
k5.metric("ROC-AUC",   f"{m['roc_auc']:.4f}")

# ── ROC Curve Image ───────────────────────────────────────────────────────────
roc_path = os.path.join(
    "models", "saved",
    f"roc_curve_{model_key.replace('_', '_')}.png"
)
if os.path.exists(roc_path):
    st.image(roc_path, caption=f"ROC Curve — {model_choice}", width=450)

st.divider()
st.caption("EMDS Prototype | KAS_004 Dissertation | For Academic Use Only")
