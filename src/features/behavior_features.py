"""
src/features/behavior_features.py
Phase 3.3 - Build behavioral features from behavioral_log.parquet.

Output:
    data/processed/behavior_features.parquet   (customer_id, ~35 features)
    outputs/reports/phase3_3_behavior.txt

Run from SamriddhiAI/ root:
    py -3.11 -m src.features.behavior_features
"""

from pathlib import Path
from datetime import datetime
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
REPORT_DIR = Path("outputs/reports")

PRODUCTS = ["personal", "home", "auto", "mortgage"]
ANALYSIS_END = pd.Timestamp("2025-06-30")


class Log:
    def __init__(self):
        self.lines = []
    def write(self, msg=""):
        print(msg)
        self.lines.append(str(msg))
    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines), encoding="utf-8")


def load_behavior(log):
    df = pd.read_parquet(INTERIM / "behavioral_log.parquet")
    log.write(f"Loaded behavioral_log: {df.shape}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ---------------------------------------------------------------
# Event counts
# ---------------------------------------------------------------
def event_counts(df, log):
    log.write("\n" + "=" * 60)
    log.write("Event counts")
    log.write("=" * 60)

    pivot = df.pivot_table(
        index="customer_id",
        columns="event_type",
        values="event_id",
        aggfunc="count",
        fill_value=0,
    ).rename(columns={
        "app_login": "login_count",
        "loan_calculator_used": "calculator_count",
        "product_page_view": "page_view_count",
        "customer_care_chat": "chat_count",
        "sms_click": "sms_click_count",
    })

    # Ensure all expected columns exist even if some event types are missing
    for col in ["login_count", "calculator_count", "page_view_count",
                "chat_count", "sms_click_count"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["total_events"] = pivot[["login_count", "calculator_count",
                                   "page_view_count", "chat_count",
                                   "sms_click_count"]].sum(axis=1)

    log.write(f"Event counts shape: {pivot.shape}")
    return pivot.reset_index()


# ---------------------------------------------------------------
# Login-specific features
# ---------------------------------------------------------------
def login_features(df, log):
    log.write("\n" + "=" * 60)
    log.write("Login features")
    log.write("=" * 60)

    logins = df[df["event_type"] == "app_login"]
    agg = logins.groupby("customer_id").agg(
        login_days_active=("timestamp", lambda s: s.dt.date.nunique()),
        first_login=("timestamp", "min"),
        last_login=("timestamp", "max"),
    ).reset_index()

    agg["login_days_since_first"] = (agg["last_login"] - agg["first_login"]).dt.days
    agg["login_days_since_last"] = (ANALYSIS_END - agg["last_login"]).dt.days
    agg["login_freq_per_month"] = agg["login_days_active"] / 6.0

    # Regularity: fraction of days between first and last login on which they logged in
    agg["login_regularity"] = (
        agg["login_days_active"] /
        (agg["login_days_since_first"] + 1)
    ).clip(upper=1.0)

    agg = agg.drop(columns=["first_login", "last_login"])

    log.write(f"Login features shape: {agg.shape}")
    log.write(f"login_days_active mean: {agg['login_days_active'].mean():.2f}")
    log.write(f"login_regularity mean: {agg['login_regularity'].mean():.4f}")
    return agg


# ---------------------------------------------------------------
# Product interest
# ---------------------------------------------------------------
def product_interest(df, log):
    log.write("\n" + "=" * 60)
    log.write("Product interest")
    log.write("=" * 60)

    # page views
    pv = df[df["event_type"] == "product_page_view"]
    pv_wide = pv.pivot_table(
        index="customer_id",
        columns="product_viewed",
        values="event_id",
        aggfunc="count",
        fill_value=0,
    ).add_prefix("pv_")

    # calculator uses
    calc = df[df["event_type"] == "loan_calculator_used"]
    calc_wide = calc.pivot_table(
        index="customer_id",
        columns="product_viewed",
        values="event_id",
        aggfunc="count",
        fill_value=0,
    ).add_prefix("calc_")

    interest = pv_wide.join(calc_wide, how="outer").fillna(0)

    # Only keep columns for our 4 products
    for prod in PRODUCTS:
        for pref in ["pv_", "calc_"]:
            col = f"{pref}{prod}"
            if col not in interest.columns:
                interest[col] = 0

    interest = interest[[f"pv_{p}" for p in PRODUCTS] +
                        [f"calc_{p}" for p in PRODUCTS]]

    # Combined interest per product
    for prod in PRODUCTS:
        interest[f"interest_{prod}"] = interest[f"pv_{prod}"] + interest[f"calc_{prod}"]

    interest["total_product_interest"] = interest[
        [f"interest_{p}" for p in PRODUCTS]
    ].sum(axis=1)

    # Shares
    for prod in PRODUCTS:
        interest[f"interest_share_{prod}"] = (
            interest[f"interest_{prod}"] /
            (interest["total_product_interest"] + 1)
        )

    # Dominant product (categorical string)
    def dominant(row):
        vals = {p: row[f"interest_{p}"] for p in PRODUCTS}
        mx = max(vals.values())
        if mx == 0:
            return "none"
        return max(vals, key=vals.get)
    interest["dominant_product"] = interest.apply(dominant, axis=1)

    log.write(f"Product interest shape: {interest.shape}")
    log.write(f"dominant_product value counts:\n"
              f"{interest['dominant_product'].value_counts()}")
    return interest.reset_index()


# ---------------------------------------------------------------
# Session features
# ---------------------------------------------------------------
def session_features(df, log):
    log.write("\n" + "=" * 60)
    log.write("Session features")
    log.write("=" * 60)

    sess = df[df["session_duration_seconds"].notna()]
    agg = sess.groupby("customer_id")["session_duration_seconds"].agg(
        total_session_seconds="sum",
        avg_session_seconds="mean",
        median_session_seconds="median",
        max_session_seconds="max",
        session_count="count",
    ).reset_index()

    log.write(f"Session features shape: {agg.shape}")
    log.write(f"avg_session_seconds mean: {agg['avg_session_seconds'].mean():.2f}")
    return agg


# ---------------------------------------------------------------
# Engagement score & funnel
# ---------------------------------------------------------------
def engagement_features(df, log):
    log.write("\n" + "=" * 60)
    log.write("Engagement score & funnel")
    log.write("=" * 60)

    # Event counts by type
    counts = df.groupby(["customer_id", "event_type"]).size() \
               .unstack(fill_value=0)

    for col in ["app_login", "loan_calculator_used", "product_page_view",
                "customer_care_chat", "sms_click"]:
        if col not in counts.columns:
            counts[col] = 0

    # avg session duration per customer
    sess_mean = df.groupby("customer_id")["session_duration_seconds"].mean().fillna(0)
    counts["avg_sess"] = sess_mean

    # Weighted log engagement
    counts["engagement_score"] = (
        0.20 * np.log1p(counts["app_login"]) +
        0.25 * np.log1p(counts["loan_calculator_used"]) +
        0.20 * np.log1p(counts["product_page_view"]) +
        0.15 * np.log1p(counts["customer_care_chat"]) +
        0.10 * np.log1p(counts["sms_click"]) +
        0.10 * np.log1p(counts["avg_sess"])
    )

    # Funnel — percentile-based on calculator_count for meaningful segmentation
    def funnel(row):
        calc = row["loan_calculator_used"]
        login = row["app_login"]
        if calc >= 8:
            return "application_intent"
        if calc >= 2:
            return "browsing"
        if login > 0:
            return "logged_in"
        return "inactive"

    counts["funnel_stage"] = counts.apply(funnel, axis=1)

    out = counts[["engagement_score", "funnel_stage"]].reset_index()

    log.write(f"Engagement features shape: {out.shape}")
    log.write(f"engagement_score describe:\n{out['engagement_score'].describe()}")
    log.write(f"funnel_stage value counts:\n{out['funnel_stage'].value_counts()}")

    return out 


# ---------------------------------------------------------------
# Assemble everything
# ---------------------------------------------------------------
def assemble(all_customers, parts, log):
    log.write("\n" + "=" * 60)
    log.write("Assembling behavior_features")
    log.write("=" * 60)

    out = all_customers.copy()

    for part in parts:
        out = out.merge(part, on="customer_id", how="left")

    # Fill numeric nulls with 0 (means "no activity of that type")
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    out[numeric_cols] = out[numeric_cols].fillna(0)

    # Categorical nulls
    out["dominant_product"] = out["dominant_product"].fillna("none")
    out["funnel_stage"] = out["funnel_stage"].fillna("inactive")

    # Flags
    out["has_behavior"] = (out["total_events"] > 0).astype(int)
    q75 = out["engagement_score"].quantile(0.75)
    out["is_highly_engaged"] = (out["engagement_score"] >= q75).astype(int)

    # One-hot funnel
    for stage in ["application_intent", "browsing", "logged_in", "inactive"]:
        out[f"funnel_{stage}"] = (out["funnel_stage"] == stage).astype(int)

    # One-hot dominant product
    for prod in PRODUCTS + ["none"]:
        out[f"dominant_{prod}"] = (out["dominant_product"] == prod).astype(int)

    log.write(f"Final shape: {out.shape}")
    log.write(f"Nulls remaining:\n{out.isnull().sum()[out.isnull().sum() > 0]}")
    return out


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log = Log()
    log.write(f"Phase 3.3 — Behavior Features — {datetime.now().isoformat(timespec='seconds')}")

    behavior = load_behavior(log)
    profile = pd.read_parquet(PROCESSED / "profile_features.parquet")
    all_customers = profile[["customer_id"]].copy()

    # Compute parts
    ec = event_counts(behavior, log)
    lf = login_features(behavior, log)
    pi = product_interest(behavior, log)
    sf = session_features(behavior, log)
    ef = engagement_features(behavior, log)

    # Assemble
    out = assemble(all_customers, [ec, lf, pi, sf, ef], log)

    out.to_parquet(PROCESSED / "behavior_features.parquet", index=False)
    log.write(f"\nSaved: {PROCESSED / 'behavior_features.parquet'}")

    log.save(REPORT_DIR / "phase3_3_behavior.txt")
    print(f"\nReport saved to {REPORT_DIR / 'phase3_3_behavior.txt'}")


if __name__ == "__main__":
    main()