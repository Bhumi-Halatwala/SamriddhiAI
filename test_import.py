"""
test_import.py
Minimal smoke test: can we import the main libs and read one CSV?
Run from SamriddhiAI/ : python test_import.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

print("Imports successful.")

df = pd.read_csv("data/raw/customer_master.csv")
print(f"\ncustomer_master shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:\n{df.head(3)}")

labels = pd.read_csv("data/raw/labels.csv")
print(f"\nlabels shape: {labels.shape}")
print(f"Columns: {list(labels.columns)}")
print(f"\napplied_flag mean: {labels['applied_flag'].mean():.4f}")
print(f"converted_flag mean: {labels['converted_flag'].mean():.4f}")

ledger = pd.read_csv("data/raw/transaction_ledger.csv")
print(f"\ntransaction_ledger shape: {ledger.shape}")
print(f"Columns: {list(ledger.columns)}")
print(f"Date range: {ledger['date'].min()} to {ledger['date'].max()}")
print(f"Avg txns per customer: {ledger.groupby('customer_id').size().mean():.1f}")