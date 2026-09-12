"""
check_setup.py
Quick sanity check after Phase 0 setup.
Verifies Python version, installed packages, folder structure, and CSV presence.
Run from SamriddhiAI/ : python -m src.data.check_setup
"""

import sys
from pathlib import Path

# ---- 1. Python version ----
print("=" * 60)
print(f"Python version: {sys.version.split()[0]}")
if sys.version_info < (3, 9):
    print("  [WARN] Python 3.9+ recommended. Some packages may fail.")
else:
    print("  [OK] Python version is fine.")

# ---- 2. Required packages ----
print("\nChecking packages...")
REQUIRED = [
    "pandas", "numpy", "sklearn", "lightgbm", "shap",
    "imblearn", "streamlit", "fastapi", "uvicorn",
    "joblib", "matplotlib", "seaborn", "pyarrow",
]
missing = []
for pkg in REQUIRED:
    try:
        __import__(pkg)
        print(f"  [OK] {pkg}")
    except ImportError:
        print(f"  [MISSING] {pkg}")
        missing.append(pkg)

if missing:
    print(f"\n  Install missing packages: pip install {' '.join(missing)}")

# ---- 3. Folder structure ----
print("\nChecking folders...")
EXPECTED_FOLDERS = [
    "data/raw", "data/interim", "data/processed",
    "notebooks", "src", "src/data", "src/features",
    "src/models", "src/chatbot", "src/guardrails", "src/api",
    "app", "models", "docs", "tests", "outputs",
]
root = Path.cwd()
for folder in EXPECTED_FOLDERS:
    path = root / folder
    status = "[OK]" if path.is_dir() else "[MISSING]"
    print(f"  {status} {folder}")

# ---- 4. CSV files ----
print("\nChecking raw CSVs...")
EXPECTED_CSVS = [
    "data/raw/customer_master.csv",
    "data/raw/transaction_ledger.csv",
    "data/raw/behavioral_log.csv",
    "data/raw/labels.csv",
]
for csv in EXPECTED_CSVS:
    path = root / csv
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  [OK] {csv}  ({size_mb:.2f} MB)")
    else:
        print(f"  [MISSING] {csv}")

print("\n" + "=" * 60)
print("Setup check complete.")
print("=" * 60)