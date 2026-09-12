"""
src/models/narration_clf.py
Phase 2 - Train a classifier that maps narration_raw -> category.

Merges 'window_dressing' into 'suspicious' (only 150 rows in original).
Trains TF-IDF + LogisticRegression and TF-IDF + LightGBM; keeps the better one.

Run from SamriddhiAI/ root:
    py -3.11 -m src.models.narration_clf

Outputs:
    models/narration_clf.pkl                (bundle: model + vectorizer + labels)
    outputs/figures/narration_confusion.png
    outputs/reports/phase2_narration.txt
"""

from pathlib import Path
from datetime import datetime
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
)
from sklearn.pipeline import Pipeline
import lightgbm as lgb

sns.set_style("whitegrid")

INTERIM = Path("data/interim")
MODELS_DIR = Path("models")
FIG_DIR = Path("outputs/figures")
REPORT_DIR = Path("outputs/reports")

RANDOM_STATE = 42
SAMPLE_SIZE = 200_000
MIN_CLASS_FOR_MERGE = 500   # categories with < this many rows get merged into 'suspicious'


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
# 1. Load + prep
# ---------------------------------------------------------------
def load_and_prepare(log):
    log.write("=" * 60)
    log.write("PHASE 2 - NARRATION PARSER")
    log.write("=" * 60)

    df = pd.read_parquet(INTERIM / "transaction_ledger.parquet",
                         columns=["narration_raw", "true_category"])
    log.write(f"\nLoaded {len(df):,} transactions")

    # Basic cleanup of narration
    df = df.dropna(subset=["narration_raw", "true_category"]).copy()
    df["narration_raw"] = df["narration_raw"].astype(str)

    # Merge rare categories into 'suspicious'
    counts = df["true_category"].value_counts()
    log.write(f"\nOriginal category counts:\n{counts}")

    rare = counts[counts < MIN_CLASS_FOR_MERGE].index.tolist()
    if rare:
        log.write(f"\nMerging rare categories into 'suspicious': {rare}")
        df["category"] = df["true_category"].replace({c: "suspicious" for c in rare})
    else:
        df["category"] = df["true_category"]

    log.write(f"\nFinal category counts:\n{df['category'].value_counts()}")
    return df


# ---------------------------------------------------------------
# 2. Stratified sample
# ---------------------------------------------------------------
def stratified_sample(df, n, log):
    if n >= len(df):
        log.write(f"\nUsing full dataset ({len(df):,} rows)")
        return df

    # Take n rows stratified by category
    frac = n / len(df)
    sampled = (
        df.groupby("category", group_keys=False)
          .apply(lambda g: g.sample(n=max(1, int(len(g) * frac)), random_state=RANDOM_STATE))
          .reset_index(drop=True)
    )
    log.write(f"\nStratified sample: {len(sampled):,} rows")
    log.write(f"Sample distribution:\n{sampled['category'].value_counts()}")
    return sampled


# ---------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------
def split(df, log):
    X = df["narration_raw"].values
    y = df["category"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    log.write(f"\nTrain size: {len(X_train):,}  Test size: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------
# 4. Train both models
# ---------------------------------------------------------------
def train_logreg(X_train, y_train):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_lgbm(X_train, y_train):
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        sublinear_tf=True,
    )
    X_train_tfidf = vec.fit_transform(X_train)
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=63,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_train_tfidf, y_train)
    return vec, clf


# ---------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------
def evaluate(name, y_true, y_pred, log):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    log.write(f"\n[{name}]")
    log.write(f"  Accuracy:     {acc:.4f}")
    log.write(f"  F1 macro:     {f1_macro:.4f}")
    log.write(f"  F1 weighted:  {f1_weighted:.4f}")
    log.write(f"\nPer-class report:\n{classification_report(y_true, y_pred, digits=4)}")
    return f1_macro, y_pred


# ---------------------------------------------------------------
# 6. Confusion matrix plot
# ---------------------------------------------------------------
def plot_confusion(y_true, y_pred, labels, log):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0], cbar=False)
    axes[0].set_title("Confusion matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=labels, yticklabels=labels, ax=axes[1], cbar=False)
    axes[1].set_title("Confusion matrix (row-normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "narration_confusion.png", dpi=110)
    plt.close()
    log.write(f"\nConfusion matrix saved to {FIG_DIR / 'narration_confusion.png'}")


# ---------------------------------------------------------------
# 7. Rule-based fallback
# ---------------------------------------------------------------
RULES = [
    ("salary",           [r"\bsal\b", r"salary", r"sal/"]),
    ("emi",              [r"\bemi\b", r"loan\d", r"ecs-emi"]),
    ("rent",             [r"rent", r"landlord"]),
    ("bounce",           [r"insufficient", r"return", r"bounce"]),
    ("business_income",  [r"business", r"receipt", r"gst"]),
    ("suspicious",       [r"transfer-in", r"window", r"round.?trip"]),
    ("discretionary",    [r"swiggy", r"zomato", r"bigbasket", r"amazon", r"flipkart", r"payment"]),
]


def rule_based(narration: str) -> str | None:
    s = narration.lower()
    for label, patterns in RULES:
        for p in patterns:
            if re.search(p, s):
                return label
    return None


def demo_rules(log, sample_texts):
    log.write("\n" + "=" * 60)
    log.write("RULE-BASED FALLBACK DEMO")
    log.write("=" * 60)
    for t in sample_texts:
        log.write(f"  {t[:60]:<62} -> {rule_based(t)}")


# ---------------------------------------------------------------
# 8. Sanity test on unseen narrations
# ---------------------------------------------------------------
UNSEEN = [
    "NEFT/XYZ/SAL/ACME-CORP",
    "ECS-EMI-BANK-HOMELOAN-2025",
    "UPI-SWIGGY-PAYMENT-4512",
    "ECS-RETURN-INSUFFICIENT FUNDS",
    "NEFT-TRANSFER-IN-300625",
]


def sanity_unseen(predict_fn, log):
    log.write("\n" + "=" * 60)
    log.write("SANITY TEST ON UNSEEN NARRATIONS")
    log.write("=" * 60)
    for t in UNSEEN:
        log.write(f"  {t:<45} -> {predict_fn(t)}")


# ---------------------------------------------------------------
# 9. Save model bundle
# ---------------------------------------------------------------
def save_bundle(kind, model, vectorizer, labels, log):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "kind": kind,                # 'logreg' or 'lgbm'
        "model": model,
        "vectorizer": vectorizer,    # None if kind=='logreg' (pipeline contains it)
        "labels": labels,
    }
    joblib.dump(bundle, MODELS_DIR / "narration_clf.pkl")
    log.write(f"\nSaved model bundle to {MODELS_DIR / 'narration_clf.pkl'}")


def load_bundle():
    return joblib.load(MODELS_DIR / "narration_clf.pkl")


def predict_bundle(bundle, texts):
    """Predict on a list of narration strings using a loaded bundle."""
    if bundle["kind"] == "logreg":
        return bundle["model"].predict(texts)
    else:
        X = bundle["vectorizer"].transform(texts)
        return bundle["model"].predict(X)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log = Log()
    log.write(f"Phase 2 Narration Parser — {datetime.now().isoformat(timespec='seconds')}")

    df = load_and_prepare(log)
    df = stratified_sample(df, SAMPLE_SIZE, log)
    X_train, X_test, y_train, y_test = split(df, log)

    labels = sorted(df["category"].unique())
    log.write(f"\nLabel set: {labels}")

    # ----- Logistic Regression -----
    log.write("\n" + "-" * 60)
    log.write("Training Logistic Regression (TF-IDF)...")
    log.write("-" * 60)
    lr_pipe = train_logreg(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    lr_f1, _ = evaluate("Logistic Regression", y_test, lr_pred, log)

    # ----- LightGBM -----
    log.write("\n" + "-" * 60)
    log.write("Training LightGBM (TF-IDF)...")
    log.write("-" * 60)
    lgb_vec, lgb_clf = train_lgbm(X_train, y_train)
    X_test_tfidf = lgb_vec.transform(X_test)
    lgb_pred = lgb_clf.predict(X_test_tfidf)
    lgb_f1, _ = evaluate("LightGBM", y_test, lgb_pred, log)

    # ----- Pick winner -----
    if lgb_f1 >= lr_f1:
        winner = "lgbm"
        winner_pred = lgb_pred
        save_bundle("lgbm", lgb_clf, lgb_vec, labels, log)
        log.write(f"\nWinner: LightGBM (F1 macro {lgb_f1:.4f} vs LR {lr_f1:.4f})")
    else:
        winner = "logreg"
        winner_pred = lr_pred
        save_bundle("logreg", lr_pipe, None, labels, log)
        log.write(f"\nWinner: Logistic Regression (F1 macro {lr_f1:.4f} vs LGBM {lgb_f1:.4f})")

    plot_confusion(y_test, winner_pred, labels, log)

    # ----- Rule demo + sanity -----
    demo_rules(log, UNSEEN)
    bundle = load_bundle()
    sanity_unseen(lambda t: predict_bundle(bundle, [t])[0], log)

    # ----- Final summary -----
    log.write("\n" + "=" * 60)
    log.write("PHASE 2 SUMMARY")
    log.write("=" * 60)
    log.write(f"Winner: {winner}")
    log.write(f"Best F1 macro: {max(lr_f1, lgb_f1):.4f}")
    log.write(f"Model bundle: models/narration_clf.pkl")
    log.write(f"Confusion matrix: outputs/figures/narration_confusion.png")

    log.save(REPORT_DIR / "phase2_narration.txt")
    print(f"\nReport saved to {REPORT_DIR / 'phase2_narration.txt'}")


if __name__ == "__main__":
    main()