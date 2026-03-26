"""
prepare_dataset.py
------------------
Downloads the EMBER 2018 dataset, cleans it, and creates
chronological train / validation / test splits.

Splits:  70% train | 15% validation | 15% test  (chronological order)
Output:  data/processed/  ->  X_train.pkl, X_val.pkl, X_test.pkl
                               y_train.pkl, y_val.pkl, y_test.pkl
                               feature_names.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
import ember
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (remainder)


def download_ember():
    """Download EMBER dataset if not already present."""
    if not os.path.exists(os.path.join(RAW_DIR, "X_train.dat")):
        print("[INFO] Downloading EMBER 2018 dataset...")
        ember.download_data(RAW_DIR)
    else:
        print("[INFO] EMBER dataset already downloaded.")


def load_ember_features():
    """Load EMBER feature matrix and labels."""
    print("[INFO] Loading EMBER features...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = ember.read_vectorized_features(
        RAW_DIR, feature_version=2
    )

    # Combine full dataset for chronological re-splitting
    X_all = np.concatenate([X_train_raw, X_test_raw], axis=0)
    y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)

    print(f"[INFO] Total samples loaded: {X_all.shape[0]:,}")
    print(f"[INFO] Features per sample:  {X_all.shape[1]:,}")
    return X_all, y_all


def clean_dataset(X, y):
    """
    Remove:
      - Unlabelled samples (y == -1)
      - Rows containing NaN or Inf values
    """
    print("[INFO] Cleaning dataset...")
    mask_labelled = y != -1
    X, y = X[mask_labelled], y[mask_labelled]
    print(f"  Removed {(~mask_labelled).sum():,} unlabelled samples.")

    mask_finite = np.isfinite(X).all(axis=1)
    removed_inf = (~mask_finite).sum()
    X, y = X[mask_finite], y[mask_finite]
    print(f"  Removed {removed_inf:,} samples with NaN/Inf values.")

    print(f"  Clean dataset size: {X.shape[0]:,} samples.")
    return X, y


def chronological_split(X, y):
    """
    Split dataset in order (preserving temporal sequence).
    EMBER samples are stored in chronological order within the raw files.
    """
    n = len(y)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, y_train = X[:train_end],        y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],          y[val_end:]

    print(f"[INFO] Split sizes:")
    print(f"  Train:      {X_train.shape[0]:>8,} samples  ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Validation: {X_val.shape[0]:>8,} samples  ({VAL_RATIO*100:.0f}%)")
    print(f"  Test:       {X_test.shape[0]:>8,} samples  ({(1-TRAIN_RATIO-VAL_RATIO)*100:.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_feature_names():
    """Return EMBER feature names (2,381 features)."""
    extractor = ember.PEFeatureExtractor(feature_version=2)
    names = []
    for feature_group in extractor.features:
        names += feature_group.feature_names()
    return names


def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    """Persist all splits to disk as pickle files."""
    artefacts = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_names": feature_names
    }
    for name, obj in artefacts.items():
        path = os.path.join(PROCESSED_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Saved: {path}")


def print_class_balance(y_train, y_val, y_test):
    """Print malicious/benign ratio per split."""
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        malicious = y.sum()
        benign    = len(y) - malicious
        print(f"  {name}: {malicious:,} malicious | {benign:,} benign  "
              f"({malicious/len(y)*100:.1f}% malicious)")


if __name__ == "__main__":
    print("=" * 60)
    print("  EMDS — Dataset Preparation")
    print("=" * 60)

    download_ember()
    X_all, y_all       = load_ember_features()
    X_all, y_all       = clean_dataset(X_all, y_all)
    X_tr, X_v, X_te, \
    y_tr, y_v, y_te    = chronological_split(X_all, y_all)

    print("\n[INFO] Class balance per split:")
    print_class_balance(y_tr, y_v, y_te)

    print("\n[INFO] Retrieving feature names...")
    try:
        feature_names = get_feature_names()
        print(f"  Feature names retrieved: {len(feature_names)}")
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_tr.shape[1])]
        print("  Using generic feature names (ember extractor unavailable).")

    print("\n[INFO] Saving processed splits...")
    save_splits(X_tr, X_v, X_te, y_tr, y_v, y_te, feature_names)

    print("\n[DONE] Dataset preparation complete.")
    print(f"       Processed files saved to: {PROCESSED_DIR}/")
