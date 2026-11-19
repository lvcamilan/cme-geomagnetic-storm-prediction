import os

# Prevent OpenMP/BLAS thread deadlocks across PyTorch, NumPy, and scikit-learn
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import copy
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# --- CONFIGURATION ---
SEED = 42
DEFAULT_PATH = Path("cme_events_2year.csv")
LEGACY_PATH = Path.home() / "Desktop" / "ML2 PROJECT" / "archive" / "cme_events_2year.csv"
DATA_PATH = DEFAULT_PATH if DEFAULT_PATH.exists() else LEGACY_PATH
TIME_COL = "start_time"
TARGET = "target_gst"
BASE_FEATURES = ["speed", "half_angle", "latitude", "longitude"]
TRAIN_FRAC = 0.80


class TorchMLP(BaseEstimator, ClassifierMixin):
    """Dense NN (Linear-ReLU-Dropout) trained with weighted BCE and chronological early stopping."""

    def __init__(
        self,
        hidden=(32, 16),
        dropout=0.3,
        lr=1e-3,
        epochs=200,
        patience=15,
        val_frac=0.15,
        batch_size=64,
        seed=SEED,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.val_frac = val_frac
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])

        # Chronological validation block from end of training data (no future leakage)
        cut = int(len(X_arr) * (1.0 - self.val_frac))
        Xtr = torch.tensor(X_arr[:cut])
        ytr = torch.tensor(y_arr[:cut]).unsqueeze(1)
        Xv = torch.tensor(X_arr[cut:])
        yv = torch.tensor(y_arr[cut:]).unsqueeze(1)

        pos_w = (ytr == 0).sum().item() / max((ytr == 1).sum().item(), 1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))

        layers, in_dim = [], X_arr.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(self.dropout)]
            in_dim = h
        self.net_ = nn.Sequential(*layers, nn.Linear(in_dim, 1))
        optimizer = torch.optim.Adam(
            self.net_.parameters(), lr=self.lr, weight_decay=1e-4
        )

        best_loss, best_state, wait = np.inf, None, 0
        for _ in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i : i + self.batch_size]
                optimizer.zero_grad()
                loss_fn(self.net_(Xtr[idx]), ytr[idx]).backward()
                optimizer.step()

            self.net_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net_(Xv), yv).item()

            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_state = copy.deepcopy(self.net_.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.net_.eval()
        with torch.no_grad():
            probs = (
                torch.sigmoid(self.net_(torch.tensor(np.asarray(X, dtype=np.float32))))
                .numpy()
                .ravel()
            )
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_models(pos_weight):
    """Create fresh preprocessing + model pipelines fit strictly inside each split."""
    return {
        "LogReg": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=SEED
            ),
        ),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED,
            ),
        ),
        "XGBoost": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight,
                eval_metric="aucpr",
                n_jobs=-1,
                random_state=SEED,
            ),
        ),
        "NeuralNet": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            TorchMLP(seed=SEED),
        ),
    }


def evaluate_probs(name, y_true, y_proba, threshold=0.5):
    """Print confusion matrix, classification report, and PR-AUC."""
    y_pred = (y_proba >= threshold).astype(int)
    print(f"\n{'=' * 10} {name} {'=' * 10}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    pr_auc = average_precision_score(y_true, y_proba)
    print(f"PR-AUC: {pr_auc:.3f} (random baseline: {np.mean(y_true):.3f})")
    return pr_auc


def main():
    # 1. Load and sort chronologically by event start time
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    df = (
        df.dropna(subset=[TIME_COL])
        .sort_values(TIME_COL, kind="stable")
        .reset_index(drop=True)
    )

    # 2. Define non-leaky target from linked_events ('GST' = geomagnetic storm impact)
    df[TARGET] = (
        df["linked_events"]
        .fillna("")
        .astype(str)
        .str.contains("GST", case=False, regex=False)
        .astype(int)
    )

    # 3. Engineer physically & temporally grounded features (past-only for rolling window)
    lat = np.radians(df["latitude"])
    lon = np.radians(df["longitude"])
    df["earth_alignment_angle"] = np.degrees(
        np.arccos(np.clip(np.cos(lat) * np.cos(lon), -1.0, 1.0))
    )

    ts = df.set_index(TIME_COL)
    df["prev_24h_cme_count"] = (
        ts["speed"].notna().astype(float).rolling("24h", closed="left").sum().values
    )
    df["prev_24h_max_speed"] = (
        ts["speed"].rolling("24h", closed="left").max().fillna(0.0).values
    )

    new_features = BASE_FEATURES + [
        "earth_alignment_angle",
        "prev_24h_cme_count",
        "prev_24h_max_speed",
    ]

    # 4. Strict chronological 80/20 split without shuffling or splitting tied timestamps
    split_idx = int(len(df) * TRAIN_FRAC)
    boundary_time = df.loc[split_idx, TIME_COL]
    split_idx = int(df[TIME_COL].lt(boundary_time).sum())

    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    assert train_df[TIME_COL].max() < test_df[TIME_COL].min(), "Temporal overlap!"
    print(
        f"TRAIN: {train_df[TIME_COL].min()} -> {train_df[TIME_COL].max()} | n={len(train_df)}"
    )
    print(
        f"TEST : {test_df[TIME_COL].min()} -> {test_df[TIME_COL].max()} | n={len(test_df)}"
    )

    def run_feature_set(features, verbose=False):
        Xtr, ytr = train_df[features], train_df[TARGET]
        Xte, yte = test_df[features], test_df[TARGET]
        pos_w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        models = build_models(pos_w)

        fitted = {}
        proba = {}
        for name, pipe in models.items():
            pipe.fit(Xtr, ytr)
            fitted[name] = pipe
            proba[name] = pipe.predict_proba(Xte)[:, 1]

        proba["Ensemble"] = np.mean(
            [proba["XGBoost"], proba["RandomForest"], proba["NeuralNet"]], axis=0
        )

        scores = {}
        for name, p in proba.items():
            if verbose:
                scores[name] = evaluate_probs(name, yte, p)
            else:
                scores[name] = average_precision_score(yte, p)
        return scores, fitted, (Xte, yte)

    # 5. Full evaluation on BASE_FEATURES + before/after PR-AUC comparison
    before_scores, xgb_base_pipe, _ = run_feature_set(BASE_FEATURES, verbose=True)
    after_scores, xgb_new_pipe, _ = run_feature_set(new_features, verbose=False)

    comparison = pd.DataFrame(
        {"PR-AUC before": before_scores, "PR-AUC after": after_scores}
    )
    comparison["delta"] = comparison["PR-AUC after"] - comparison["PR-AUC before"]
    print("\n=== PR-AUC Before vs. After Feature Engineering ===")
    print(comparison.round(3).to_string())

    # 6. Feature importance (native XGBoost + test-set Permutation Importance on PR-AUC)
    Xte_base, yte_test = test_df[BASE_FEATURES], test_df[TARGET]
    Xte_new = test_df[new_features]

    for label, pipe, X_eval, feat_list in [
        ("BASE_FEATURES", xgb_base_pipe["XGBoost"], Xte_base, BASE_FEATURES),
        ("NEW_FEATURES", xgb_new_pipe["XGBoost"], Xte_new, new_features),
    ]:
        perm = permutation_importance(
            pipe,
            X_eval,
            yte_test,
            scoring="average_precision",
            n_repeats=20,
            random_state=SEED,
            n_jobs=-1,
        )
        xgb_est = pipe.named_steps["xgbclassifier"]
        imp_df = pd.DataFrame(
            {
                "feature": feat_list,
                "xgb_importance": xgb_est.feature_importances_,
                "perm_pr_auc_mean": perm.importances_mean,
                "perm_pr_auc_std": perm.importances_std,
            }
        ).sort_values("perm_pr_auc_mean", ascending=False)
        print(f"\n=== XGBoost Feature Importance [{label}] ===")
        print(imp_df.round(4).to_string(index=False))

    # Plot side-by-side permutation importance
    perm_b = permutation_importance(
        xgb_base_pipe["XGBoost"],
        Xte_base,
        yte_test,
        scoring="average_precision",
        n_repeats=20,
        random_state=SEED,
        n_jobs=-1,
    )
    perm_n = permutation_importance(
        xgb_new_pipe["XGBoost"],
        Xte_new,
        yte_test,
        scoring="average_precision",
        n_repeats=20,
        random_state=SEED,
        n_jobs=-1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=False)
    plot_specs = [
        (axes[0], BASE_FEATURES, perm_b, "Base Features (4)"),
        (axes[1], new_features, perm_n, "Engineered Features (7)"),
    ]
    for ax, feats, res, title in plot_specs:
        order = np.argsort(res.importances_mean)
        ax.barh(
            np.array(feats)[order],
            res.importances_mean[order],
            xerr=res.importances_std[order],
            capsize=3,
        )
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Mean PR-AUC Drop on Test Set")

    plt.suptitle("XGBoost Permutation Importance (Test Set, 20 repeats)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
