"""
EDA script - run this yourself, section by section.

HOW TO RUN IN VS CODE:
  1. Put the 4 CSVs in a folder called `data/` next to this file.
  2. Install pandas: pip install pandas
  3. Right-click this file -> "Run Python File", OR better -
     use VS Code's "Run Cell" (the '# %%' lines below turn this into
     Jupyter-style cells you can run one at a time and see output inline).
  4. Go top to bottom, read the printed output before moving to the next cell.
"""

# %% 1. Load everything and check basic shape
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

customer_master = pd.read_csv("data/customer_master.csv")
labels = pd.read_csv("data/labels.csv")
behavioral_log = pd.read_csv("data/behavioral_log.csv")
transaction_ledger = pd.read_csv("data/transaction_ledger.csv")

for name, df in [
    ("customer_master", customer_master),
    ("labels", labels),
    ("behavioral_log", behavioral_log),
    ("transaction_ledger", transaction_ledger),
]:
    print(f"{name}: {df.shape[0]} rows x {df.shape[1]} cols")
    print(df.dtypes, "\n")

# %% 2. Check for missing values - decides your preprocessing plan
print("customer_master nulls:\n", customer_master.isnull().sum())
print("\nlabels nulls:\n", labels.isnull().sum())
print("\nbehavioral_log nulls:\n", behavioral_log.isnull().sum())
print("\ntransaction_ledger nulls:\n", transaction_ledger.isnull().sum())
# Question to answer: which columns have nulls, and is null meaningful
# (e.g. "no existing loan") or a data quality problem?

# %% 3. Understand customer_master - who are these customers?
print(customer_master.describe())
for col in ["occupation_type", "city_tier", "residence_type"]:
    print(f"\n{col}:\n", customer_master[col].value_counts())
# Question to answer: what's the age/income/bureau-score range?
# Are the categories balanced or skewed?

# %% 4. Understand labels - this is your target variable
print(labels.applied_flag.value_counts())
print(labels.converted_flag.value_counts())
print(labels.loan_type.value_counts())
print(pd.crosstab(labels.applied_flag, labels.converted_flag))
# Question to answer: is the dataset balanced (roughly equal applied/not-applied)?
# Note applied_flag=0 always means converted_flag=0 and loan_type='none' - makes sense,
# you can't convert without applying.

# %% 5. Understand behavioral_log - what do customers do in the app?
print(behavioral_log.event_type.value_counts())
print(behavioral_log.product_viewed.value_counts())
print("date range:", behavioral_log.timestamp.min(), "to", behavioral_log.timestamp.max())
print(behavioral_log.groupby("customer_id").size().describe())
# Question to answer: how many events per customer on average? Any customers
# with almost no activity (cold-start problem for personalization)?

# %% 6. Understand transaction_ledger - the richest file
print(transaction_ledger.true_category.value_counts())
print(transaction_ledger.direction.value_counts())
print(transaction_ledger.channel.value_counts())
print("date range:", transaction_ledger.date.min(), "to", transaction_ledger.date.max())
print(transaction_ledger.groupby("customer_id").size().describe())
# Question to answer: which categories exist? "bounce" and "window_dressing" are
# NOT obvious from a quick glance at the column list - you only find them by
# running value_counts(). This is exactly why step 6 matters.

# %% 7. Look closer at the two unusual categories - stress and fraud signals
bounce = transaction_ledger[transaction_ledger.true_category == "bounce"]
window_dressing = transaction_ledger[transaction_ledger.true_category == "window_dressing"]

print("Bounce: affects", bounce.customer_id.nunique(), "unique customers")
print(bounce.narration_raw.value_counts().head())

print("\nWindow dressing: affects", window_dressing.customer_id.nunique(), "unique customers")
print(window_dressing.narration_raw.value_counts().head())
print("Window dressing dates:\n", window_dressing.date.value_counts())
# Question to answer: when do window_dressing transactions happen relative to
# the rest of the timeline? (Hint: check if they cluster near the most recent dates.)

# %% 8. Check that customer_id keys line up across all 4 files
print("labels ids == customer_master ids:", set(labels.customer_id) == set(customer_master.customer_id))
print("transaction_ledger ids subset of customer_master:",
      set(transaction_ledger.customer_id).issubset(set(customer_master.customer_id)))
print("behavioral_log ids subset of customer_master:",
      set(behavioral_log.customer_id).issubset(set(customer_master.customer_id)))
# If any of these print False, you have an orphan-key problem to fix before merging.

# %% 9. Your own follow-up questions - add cells here
# Try: does bureau_score_proxy differ between converted and non-converted customers?
# Try: does existing_loan_count relate to applied_flag?
# Try: pick 3 random customer_ids and manually trace their full transaction history.
