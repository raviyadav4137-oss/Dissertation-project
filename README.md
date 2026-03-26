# Explainable Malware Detection System (EMDS)
### KAS_004 Dissertation Project

> **Evaluating the Effectiveness of Explainable AI (XAI) Techniques in Enhancing Malware Detection for Security Analysts**

---

## Project Overview

This repository contains the full implementation of the **Explainable Malware Detection System (EMDS)** — a dissertation project that evaluates how XAI methods (SHAP and LIME) can make ML-based malware classifiers more transparent and useful for Security Operations Centre (SOC) analysts.

The system trains machine learning classifiers on the [EMBER dataset](https://github.com/elastic/ember) and applies SHAP and LIME explanations to both global model behaviour and individual malware predictions, presented through an interactive Streamlit dashboard.

---

## Repository Structure

```
emds/
├── data/
│   └── prepare_dataset.py        # Download, clean, split EMBER dataset
├── models/
│   └── train_classifier.py       # Train XGBoost + Random Forest, evaluate metrics
├── explainability/
│   ├── shap_explainer.py         # SHAP global + local explanations
│   └── lime_explainer.py         # LIME instance-level explanations
├── evaluation/
│   └── evaluate_xai.py           # XAI evaluation framework (technical + analyst)
├── dashboard/
│   └── app.py                    # Streamlit analyst dashboard
├── utils/
│   └── helpers.py                # Shared utilities (plotting, metrics, logging)
├── notebooks/
│   └── exploration.ipynb         # Exploratory data analysis notebook
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/emds-kas004.git
cd emds-kas004
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the EMBER dataset
```bash
pip install ember
python data/prepare_dataset.py
```
> The EMBER 2018 dataset (~1.6 GB) will be downloaded and processed automatically.

---

## Running the Project

### Step 1 — Prepare the dataset
```bash
python data/prepare_dataset.py
```

### Step 2 — Train classifiers
```bash
python models/train_classifier.py
```

### Step 3 — Generate SHAP explanations
```bash
python explainability/shap_explainer.py
```

### Step 4 — Generate LIME explanations
```bash
python explainability/lime_explainer.py
```

### Step 5 — Run the evaluation framework
```bash
python evaluation/evaluate_xai.py
```

### Step 6 — Launch the analyst dashboard
```bash
streamlit run dashboard/app.py
```

---

## System Requirements

| Component       | Requirement                          |
|----------------|--------------------------------------|
| Python          | 3.9 or higher                        |
| RAM             | 16 GB minimum recommended            |
| CPU             | Multi-core (Intel i7 or equivalent)  |
| GPU             | Not required                         |
| Storage         | ~3 GB (dataset + model artefacts)    |

---

## Key Libraries

| Library       | Purpose                          |
|--------------|----------------------------------|
| `xgboost`     | Primary malware classifier       |
| `scikit-learn`| Random Forest + evaluation tools |
| `shap`        | SHAP explanations                |
| `lime`        | LIME explanations                |
| `streamlit`   | Analyst dashboard UI             |
| `pandas`      | Data manipulation                |
| `matplotlib`  | Visualisation                    |

---

## Dataset

This project uses the **EMBER 2018** dataset:
- ~1.1 million Windows PE file samples
- 2,381 pre-extracted static features per sample
- Labels: Malicious (1) / Benign (0) via VirusTotal consensus
- Reference: Anderson & Roth (2018) — [arXiv:1804.04637](https://arxiv.org/abs/1804.04637)

> No malicious binaries are downloaded or executed. EMBER provides pre-extracted feature vectors only.

---

## Dissertation Author
**Student ID:** KAS_004  
**Supervisor:** [Supervisor Name]  
**Institution:** [University Name]  
**Submission:** May 2026
