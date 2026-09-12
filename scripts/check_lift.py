"""
scripts/check_lift.py
Quick diagnostic: does the anomaly detector flag injected customers
at a higher rate than non-injected customers? (Lift metric)

Run from SamriddhiAI/ root:
    py -3.11 scripts/check_lift.py
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_INJECT = 100

print("Loading ledger...")
m = pd.read_parquet("data/interim/transaction_ledger.parquet")
m["date"] = pd.to_datetime(m["date"])
m["month"] = m["date"].dt.to_period("M").dt.to_timestamp()
m["abs_amount"] = m["amount"].abs()

print("Building monthly base...")
base = m.groupby(["customer_id", "month"]).agg(
    txn_count=("amount", "size"),
    credit_sum=("amount", lambda s: s[s > 0].sum()),
    debit_sum=("amount", lambda s: -s[s < 0].sum()),
).reset_index()

base["net_flow"] = base["credit_sum"] - base["debit_sum"]
base["savings_rate"] = base["net_flow"] / (base["credit_sum"] + 1)

# Salary + discretionary monthly
sal = m[m["true_category"] == "salary"].groupby(
    ["customer_id", "month"])["abs_amount"].sum().reset_index(name="salary_sum")
disc = m[m["true_category"] == "discretionary"].groupby(
    ["customer_id", "month"])["abs_amount"].sum().reset_index(name="discretionary_sum")

base = base.merge(sal, on=["customer_id", "month"], how="left") \
           .merge(disc, on=["customer_id", "month"], how="left").fillna(0)

# Inject anomalies into 100 customers
rng = np.random.RandomState(RANDOM_STATE)
custs = base["customer_id"].unique()
inject = rng.choice(custs, N_INJECT, replace=False)

injected_rows = 0
for cid in inject:
    sub = base[base["customer_id"] == cid].sort_values("month")
    if len(sub) < 5:
        continue
    tgt = sub.index[3:5]  # months 4 and 5
    base.loc[tgt, "salary_sum"] = 0
    base.loc[tgt, "savings_rate"] = -0.5
    base.loc[tgt, "discretionary_sum"] = base.loc[tgt, "discretionary_sum"] * 3
    injected_rows += 2

print(f"Injected {injected_rows} rows across {len(inject)} customers")

# Per-customer z-score on savings_rate
base = base.sort_values(["customer_id", "month"])
grp = base.groupby("customer_id")["savings_rate"]
mean = grp.transform("mean")
std = grp.transform("std").replace(0, np.nan)
base["z_sav"] = ((base["savings_rate"] - mean) / std).fillna(0).abs()
base["anom"] = (base["z_sav"] >= 2.5).astype(int)

per_c = base.groupby("customer_id")["anom"].sum().reset_index()
per_c["inj"] = per_c["customer_id"].isin(inject).astype(int)
per_c["flag"] = (per_c["anom"] >= 2).astype(int)

inj_flagged = per_c[per_c["inj"] == 1]["flag"].sum()
non_inj_flagged = per_c[per_c["inj"] == 0]["flag"].sum()

inj_rate = per_c[per_c["inj"] == 1]["flag"].mean()
non_inj_rate = per_c[per_c["inj"] == 0]["flag"].mean()
lift = inj_rate / non_inj_rate if non_inj_rate > 0 else float("inf")

print(f"\nInjected flagged:     {inj_flagged}/{N_INJECT} ({inj_rate:.3f})")
print(f"Non-injected flagged: {non_inj_flagged}/4900 ({non_inj_rate:.3f})")
print(f"Lift:                 {lift:.2f}x")