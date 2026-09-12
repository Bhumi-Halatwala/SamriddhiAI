"""
src/models/recommender.py
Phase 4 - Train propensity + product recommendation models.

Head A: P(apply) — binary LightGBM on applied_flag
Head B: P(product | apply) — multiclass LightGBM on loan_type (excluding 'none')

Also builds:
    - SHAP explanations on the test set
    - ROC / PR / calibration / feature-importance plots
    - A rule-based fallback recommender (for demo / sanity)

Run from SamriddhiAI/ root:
    py -3.11 -m src.models.recommender

Outputs:
    models/recommender.pkl
    models/recommender_head_a.txt          (LightGBM booster text)
    models/recommender_head_b.txt
    outputs/figures/recommender_*.png
    outputs/reports/phase4_recommender.txt
"""

from pathlib import Path
from datetime import datetime
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score,
    roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve

sns.set_style("whitegrid")
RANDOM_STATE = 42

PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")
FIG_DIR = Path("outputs/figures")
REPORT_DIR = Path("outputs/reports")

LABEL_COLS = ["applied_flag", "converted_flag", "loan_type",
              "application_amount", "has_application_label"]

PRODUCTS = ["home", "auto", "personal", "mortgage"]


class Log:
    def __init__(self):
        self.lines = []
    def write(self, msg=""):
        print(msg)
        self.lines.append(str(msg))
    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines), encoding="utf-8")


# ---------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------
def load_and_prepare(log):
    log.write("\n" + "=" * 60)
    log.write("Loading model_input + labels")
    log.write("=" * 60)

    X = pd.read_parquet(PROCESSED / "model_input.parquet")
    y = pd.read_parquet(PROCESSED / "labels_joined.parquet")

    # Drop non-numeric columns that aren't features
    drop_cols = ["customer_id", "stress_level"]
    present = [c for c in drop_cols if c in X.columns]
    X = X.drop(columns=present)
    feature_names = X.columns.tolist()

    # Sanity: labels align by row order (both were built from same customer order)
    assert len(X) == len(y), f"Length mismatch: {len(X)} vs {len(y)}"
    df = pd.concat([X, y[["applied_flag", "loan_type", "converted_flag"]]], axis=1)

    log.write(f"Features: {X.shape[1]}")
    log.write(f"Rows: {len(X)}")
    log.write(f"applied_flag mean: {df['applied_flag'].mean():.4f}")
    log.write(f"loan_type value counts:\n{df['loan_type'].value_counts()}")

    return df, feature_names


def split_data(df, feature_names, log, strat_col="applied_flag"):
    log.write("\n" + "=" * 60)
    log.write(f"Splitting (70/15/15) on '{strat_col}'")
    log.write("=" * 60)

    X = df[feature_names]
    y = df[strat_col]

    # First split off test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y,
    )
    # Then split train/val from remaining
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.176, random_state=RANDOM_STATE, stratify=y_tmp,
    )

    log.write(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    log.write(f"Train applied mean: {y_train.mean():.4f}")
    log.write(f"Val   applied mean: {y_val.mean():.4f}")
    log.write(f"Test  applied mean: {y_test.mean():.4f}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------
# Head A: Propensity (P(apply))
# ---------------------------------------------------------------
def train_head_a(X_train, y_train, X_val, y_val, feature_names, log):
    log.write("\n" + "=" * 60)
    log.write("Training Head A (Propensity)")
    log.write("=" * 60)

    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(0)],
    )
    log.write(f"Best iteration: {clf.best_iteration_}")
    return clf


def evaluate_head_a(clf, X_test, y_test, log):
    log.write("\n" + "-" * 60)
    log.write("Head A — Test evaluation")
    log.write("-" * 60)

    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)

    log.write(f"AUC: {auc:.4f}")
    log.write(f"Avg precision: {ap:.4f}")
    log.write(f"Brier score: {brier:.4f}")

    # At threshold 0.5
    pred = (proba >= 0.5).astype(int)
    log.write(f"\nClassification report (threshold=0.5):")
    log.write(classification_report(y_test, pred, digits=4))
    log.write(f"Confusion matrix:\n{confusion_matrix(y_test, pred)}")

    return proba, auc, ap, brier


# ---------------------------------------------------------------
# Head B: Product (P(product | apply))
# ---------------------------------------------------------------
def train_head_b(df, feature_names, log):
    log.write("\n" + "=" * 60)
    log.write("Training Head B (Product classifier)")
    log.write("=" * 60)

    # Keep only customers who applied AND have a known product
    mask = (df["applied_flag"] == 1) & (df["loan_type"].isin(PRODUCTS))
    sub = df[mask].copy()
    log.write(f"Head B training population: {len(sub)} customers")
    log.write(f"Product distribution:\n{sub['loan_type'].value_counts()}")

    X = sub[feature_names]
    y = sub["loan_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(PRODUCTS),
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    pred = clf.classes_[np.argmax(proba, axis=1)]
    f1_macro = f1_score(y_test, pred, average="macro")

    log.write(f"\nHead B test macro-F1: {f1_macro:.4f}  (random baseline = 0.25)")
    log.write(f"\nClassification report:\n"
              f"{classification_report(y_test, pred, digits=4)}")

    return clf, f1_macro


# ---------------------------------------------------------------
# SHAP explanations
# ---------------------------------------------------------------
def shap_explain_head_a(clf, X_test, log, n_samples=500):
    log.write("\n" + "=" * 60)
    log.write("SHAP explanations (Head A)")
    log.write("=" * 60)

    # Subsample for speed
    X_sample = X_test.sample(min(n_samples, len(X_test)), random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    # Global importance
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    log.write(f"\nTop 20 global features:")
    for i, row in importance.head(20).iterrows():
        log.write(f"  {i+1:2d}. {row['feature']:<35} {row['mean_abs_shap']:.4f}")

    # Save importance CSV
    importance.to_csv(REPORT_DIR / "phase4_shap_importance.csv", index=False)

    # Beeswarm plot
    plt.figure(figsize=(9, 8))
    shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "recommender_shap_beeswarm.png", dpi=110, bbox_inches="tight")
    plt.close()
    log.write(f"\nSaved: {FIG_DIR / 'recommender_shap_beeswarm.png'}")

    return explainer, shap_values, importance


# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
def plot_roc_pr(y_test, proba, log):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    axes[0].plot(fpr, tpr, color="#264653", label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#888")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve (Head A)")
    axes[0].legend()

    # PR
    p, r, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    axes[1].plot(r, p, color="#e76f51", label=f"AP = {ap:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve (Head A)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "recommender_roc_pr.png", dpi=110)
    plt.close()
    log.write(f"Saved: {FIG_DIR / 'recommender_roc_pr.png'}")


def plot_calibration(y_test, proba, log):
    fig, ax = plt.subplots(figsize=(6, 6))
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
    ax.plot(prob_pred, prob_true, marker="o", color="#264653", label="Model")
    ax.plot([0, 1], [0, 1], "--", color="#888", label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Actual rate")
    ax.set_title("Calibration (Head A)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "recommender_calibration.png", dpi=110)
    plt.close()
    log.write(f"Saved: {FIG_DIR / 'recommender_calibration.png'}")


def plot_feature_importance(clf, feature_names, log, top_n=25):
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="#2a9d8f")
    ax.set_xlabel("Split-based importance")
    ax.set_title(f"Head A — Top {top_n} features")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "recommender_feature_importance.png", dpi=110)
    plt.close()
    log.write(f"Saved: {FIG_DIR / 'recommender_feature_importance.png'}")


# ---------------------------------------------------------------
# Rule-based fallback recommender
# ---------------------------------------------------------------
def rule_based_recommendation(row):
    """
    Deterministic rule-based recommender. Fallback if ML is weak.

    Uses interpretable business rules based on customer profile + stress level.
    Returns: (product, reason)
    """
    if row.get("is_stressed", 0) == 1:
        return ("none", "High financial stress — no loan offers")

    age = row.get("age", 35)
    has_home = row.get("has_home_loan", 0) == 1
    has_mortgage = row.get("has_mortgage_loan", 0) == 1
    loan_count = row.get("existing_loan_count", 0)
    interest_home = row.get("interest_home", 0)
    interest_auto = row.get("interest_auto", 0)
    interest_personal = row.get("interest_personal", 0)
    interest_mortgage = row.get("interest_mortgage", 0)

    # Highest shown interest wins
    interests = {
        "home": interest_home,
        "auto": interest_auto,
        "personal": interest_personal,
        "mortgage": interest_mortgage,
    }
    top_interest = max(interests, key=interests.get)
    top_interest_val = interests[top_interest]

    if top_interest_val > 0:
        return (top_interest, f"Customer showed interest in {top_interest} products")

    # Fall through to profile rules
    if age <= 30 and row.get("residence_owned", 0) == 0 and loan_count == 0:
        return ("personal", "Young renter, no loans — fits personal loan profile")
    if 30 < age <= 45 and loan_count == 0 and has_home == 0:
        return ("home", "Family-age, no home loan — fits home loan profile")
    if has_mortgage:
        return ("mortgage", "Existing mortgage holder — mortgage top-up candidate")
    if loan_count >= 1:
        return ("personal", "Existing borrower — personal loan consolidation candidate")
    return ("personal", "Default personal loan recommendation")


def demo_rule_based(df, log, n=5):
    log.write("\n" + "=" * 60)
    log.write("Rule-based fallback recommender — demo")
    log.write("=" * 60)

    sample = df.sample(n, random_state=RANDOM_STATE)
    for _, row in sample.iterrows():
        product, reason = rule_based_recommendation(row)
        log.write(f"  customer={row.get('customer_id', 'unknown')} -> "
                  f"{product}  |  {reason}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log = Log()
    log.write(f"Phase 4 — Recommender Models — {datetime.now().isoformat(timespec='seconds')}")

    df, feature_names = load_and_prepare(log)

    # ============ Head A ============
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, feature_names, log, strat_col="applied_flag"
    )
    clf_a = train_head_a(X_train, y_train, X_val, y_val, feature_names, log)
    proba_a, auc_a, ap_a, brier_a = evaluate_head_a(clf_a, X_test, y_test, log)

    plot_roc_pr(y_test, proba_a, log)
    plot_calibration(y_test, proba_a, log)
    plot_feature_importance(clf_a, feature_names, log)

    explainer_a, shap_values, importance_a = shap_explain_head_a(clf_a, X_test, log)

    # ============ Head B ============
    clf_b, f1_b = train_head_b(df, feature_names, log)

    # ============ Rule-based fallback ============
    demo_rule_based(df, log)

    # ============ Save bundle ============
    bundle = {
        "head_a": clf_a,
        "head_b": clf_b,
        "feature_names": feature_names,
        "products": PRODUCTS,
        "metrics": {
            "auc_a": float(auc_a),
            "ap_a": float(ap_a),
            "brier_a": float(brier_a),
            "f1_b": float(f1_b),
        },
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, MODELS_DIR / "recommender.pkl")
    clf_a.booster_.save_model(str(MODELS_DIR / "recommender_head_a.txt"))
    clf_b.booster_.save_model(str(MODELS_DIR / "recommender_head_b.txt"))
    log.write(f"\nSaved: {MODELS_DIR / 'recommender.pkl'}")

    # ============ Final summary ============
    log.write("\n" + "=" * 60)
    log.write("PHASE 4 SUMMARY")
    log.write("=" * 60)
    log.write(f"Head A (propensity):  AUC = {auc_a:.4f}, AP = {ap_a:.4f}, "
              f"Brier = {brier_a:.4f}")
    log.write(f"Head B (product):     macro-F1 = {f1_b:.4f}")
    log.write(f"\nTop 10 features by SHAP:")
    for i, row in importance_a.head(10).iterrows():
        log.write(f"  {i+1:2d}. {row['feature']:<35} {row['mean_abs_shap']:.4f}")

    log.write(f"\nInterpretation:")
    if auc_a >= 0.65:
        log.write("  Head A shows meaningful predictive power. "
                  "Ready for dashboard integration.")
    elif auc_a >= 0.55:
        log.write("  Head A shows weak but non-trivial signal. "
                  "Document as a limitation; rule-based fallback recommended.")
    else:
        log.write("  Head A is near-random. Use the rule-based recommender "
                  "for the demo, and document the ML limitation honestly.")

    log.save(REPORT_DIR / "phase4_recommender.txt")
    print(f"\nReport saved to {REPORT_DIR / 'phase4_recommender.txt'}")


if __name__ == "__main__":
    main()