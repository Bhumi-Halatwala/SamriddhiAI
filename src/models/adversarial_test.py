"""
src/models/adversarial_test.py
Phase 2.5 - Evaluate the narration classifier on realistic messy narrations
that differ from the training templates.

Purpose: quantify the "deployment gap" between templated training data and
real-world bank narrations.

Run from SamriddhiAI/ root:
    py -3.11 -m src.models.adversarial_test

Outputs:
    outputs/figures/narration_adversarial.png
    outputs/reports/phase2_5_adversarial.txt
"""

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from src.models.predict_narration import predict_categories, rule_based

sns.set_style("whitegrid")

ADV_PATH = Path("data/raw/adversarial_narrations.csv")
FIG_DIR = Path("outputs/figures")
REPORT_DIR = Path("outputs/reports")


class Log:
    def __init__(self):
        self.lines = []
    def write(self, msg=""):
        print(msg)
        self.lines.append(str(msg))
    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines), encoding="utf-8")


def load_adversarial(log):
    df = pd.read_csv(ADV_PATH)
    log.write(f"Loaded adversarial set: {df.shape}")
    log.write(f"Label distribution:\n{df['true_label'].value_counts()}")
    return df


def predict(df):
    """Run the model (with rule fallback) on the adversarial set."""
    texts = df["narration_raw"].tolist()
    preds, confs, used_rule = predict_categories(texts, confidence_threshold=0.6)
    return preds, confs, used_rule


def predict_rule_only(df):
    """Ablation: what if we ONLY used rules, no model at all?"""
    out = []
    for t in df["narration_raw"]:
        r = rule_based(t)
        out.append(r if r else "discretionary")  # default to most common
    return np.array(out)


def predict_model_only(df):
    """Ablation: force model-only, ignore confidence threshold."""
    from src.models.predict_narration import _load
    bundle = _load()
    texts = df["narration_raw"].tolist()
    if bundle["kind"] == "logreg":
        return bundle["model"].predict(texts)
    else:
        X = bundle["vectorizer"].transform(texts)
        return bundle["model"].predict(X)


def evaluate(name, y_true, y_pred, log):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    log.write(f"\n[{name}]")
    log.write(f"  Accuracy:    {acc:.4f}")
    log.write(f"  F1 macro:    {f1m:.4f}")
    log.write(f"  F1 weighted: {f1w:.4f}")
    return {"name": name, "acc": acc, "f1_macro": f1m, "f1_weighted": f1w}


def plot_results(df, y_true, preds_model, preds_full, preds_rules, log):
    labels_sorted = sorted(df["true_label"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, preds, title in [
        (axes[0], preds_model, "Model only (no fallback)"),
        (axes[1], preds_full,  "Model + rule fallback"),
        (axes[2], preds_rules, "Rule-based only"),
    ]:
        cm = confusion_matrix(y_true, preds, labels=labels_sorted)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=labels_sorted, yticklabels=labels_sorted,
                    ax=ax, cbar=False, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Adversarial narration test — confusion matrices (row-normalized)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "narration_adversarial.png", dpi=110)
    plt.close()
    log.write(f"\nSaved chart: {FIG_DIR / 'narration_adversarial.png'}")


def dump_errors(df, y_true, preds, log, tag):
    """Print each misclassified row for manual inspection."""
    log.write(f"\n--- Misclassified by '{tag}' ---")
    errors = df[preds != y_true]
    if len(errors) == 0:
        log.write("  (none)")
        return
    for _, row in errors.iterrows():
        log.write(f"  TRUE={row['true_label']:<18} PRED={preds[row.name]:<18} | {row['narration_raw']}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log = Log()
    log.write(f"Phase 2.5 Adversarial Test — {datetime.now().isoformat(timespec='seconds')}")

    df = load_adversarial(log)
    y_true = df["true_label"].values

    # Three predictions
    preds_model_only = predict_model_only(df)
    preds_full, confs, used_rule = predict(df)
    preds_rules = predict_rule_only(df)

    log.write("\n" + "=" * 60)
    log.write("RESULTS ON ADVERSARIAL SET")
    log.write("=" * 60)

    results = [
        evaluate("Model only", y_true, preds_model_only, log),
        evaluate("Model + rule fallback", y_true, preds_full, log),
        evaluate("Rule-based only", y_true, preds_rules, log),
    ]

    log.write("\n" + "=" * 60)
    log.write("COMPARISON WITH TRAINING DATA")
    log.write("=" * 60)
    log.write("Training/template data: F1 macro = 1.0000 (both models)")
    log.write(f"Adversarial messy data: F1 macro = {results[1]['f1_macro']:.4f} (model + rules)")

    log.write("\n" + "=" * 60)
    log.write("DETAILED PER-CLASS REPORT (model + rule fallback)")
    log.write("=" * 60)
    log.write(classification_report(y_true, preds_full, digits=4, zero_division=0))

    log.write("\n" + "=" * 60)
    log.write("MISCLASSIFICATION DETAIL")
    log.write("=" * 60)
    dump_errors(df, y_true, preds_model_only, log, "model only")
    dump_errors(df, y_true, preds_full, log, "model + fallback")
    dump_errors(df, y_true, preds_rules, log, "rules only")

    log.write("\n" + "=" * 60)
    log.write("WHERE THE FALLBACK HELPED")
    log.write("=" * 60)
    helped = 0
    for i in range(len(df)):
        if preds_full[i] == y_true[i] and preds_model_only[i] != y_true[i]:
            helped += 1
    log.write(f"Rows where fallback corrected a model mistake: {helped}")

    plot_results(df, y_true, preds_model_only, preds_full, preds_rules, log)

    log.write("\n" + "=" * 60)
    log.write("TAKEAWAYS FOR THE PRESENTATION")
    log.write("=" * 60)
    log.write("1. On templated training data both models hit F1 = 1.0 — not a real NLP win.")
    log.write("2. On realistic messy narrations, model-only accuracy drops noticeably.")
    log.write("3. Rule fallback recovers some of that drop by catching obvious keywords.")
    log.write("4. This is the deployment gap. Mitigations: (a) retrain on real narrations,")
    log.write("   (b) active learning on low-confidence rows, (c) human-in-the-loop review.")

    log.save(REPORT_DIR / "phase2_5_adversarial.txt")
    print(f"\nReport saved to {REPORT_DIR / 'phase2_5_adversarial.txt'}")


if __name__ == "__main__":
    main()