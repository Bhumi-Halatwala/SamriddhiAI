"""
Data Manager for Bharat Banking AI.
Loads and manages customer profiles from data/customer_master.csv
and transactions from data/transaction_ledger.csv.
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CUSTOMER_FILE = os.path.join(DATA_DIR, "customer_master.csv")
TRANSACTION_FILE = os.path.join(DATA_DIR, "transaction_ledger.csv")

class DataManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.customers = {}
        self.tx_cache = {}
        self._load_customers()

    def reload(self):
        """Reload customer profiles when the source dataset changes or was unavailable."""
        self.customers.clear()
        self.tx_cache.clear()
        self._load_customers()

    def _load_customers(self):
        if not os.path.exists(CUSTOMER_FILE):
            return
        df = pd.read_csv(CUSTOMER_FILE)
        for _, row in df.iterrows():
            cid = str(row["customer_id"]).strip()
            ann_inc = float(row["declared_annual_income"]) if pd.notnull(row["declared_annual_income"]) else 0.0
            bureau = int(row["bureau_score_proxy"]) if pd.notnull(row["bureau_score_proxy"]) else 650
            loan_cnt = int(row["existing_loan_count"]) if pd.notnull(row["existing_loan_count"]) else 0
            loan_types_raw = str(row["existing_loan_types"]).strip() if pd.notnull(row["existing_loan_types"]) else ""
            if loan_types_raw == "nan" or not loan_types_raw:
                loan_types = []
            else:
                loan_types = [l.strip() for l in loan_types_raw.replace("|", ",").split(",") if l.strip()]

            # Score description
            if bureau >= 750:
                bureau_grade = "Excellent / उत्कृष्ट / ઉત્કૃષ્ટ"
            elif bureau >= 700:
                bureau_grade = "Good / अच्छा / સારું"
            elif bureau >= 650:
                bureau_grade = "Fair / सामान्य / મધ્યમ"
            else:
                bureau_grade = "Needs Improvement / सुधार की आवश्यकता"

            # KYC status evaluation
            tenure = float(row["relationship_tenure_years"]) if pd.notnull(row["relationship_tenure_years"]) else 1.0
            if tenure > 5.0:
                kyc_status = "Periodic Re-KYC Due (Video KYC Available)"
                kyc_status_hi = "समय-समय पर री-केवाईसी देय है (वीडियो केवाईसी उपलब्ध)"
                kyc_status_gu = "સમયાંતરે રી-કેવાયસી બાકી છે (વિડિયો કેવાયસી ઉપલબ્ધ)"
                kyc_verified = False
            else:
                kyc_status = "Fully Verified & Active (Aadhaar & PAN Linked)"
                kyc_status_hi = "पूर्णतः सत्यापित और सक्रिय (आधार और पैन लिंक हैं)"
                kyc_status_gu = "સંપૂર્ણ ચકાસાયેલ અને સક્રિય (આધાર અને પાન લિંક છે)"
                kyc_verified = True

            # Realistic simulated balance based on annual income
            monthly_income = int(ann_inc / 12) if ann_inc > 0 else 35000
            np.random.seed(int(cid.replace("CUST", "")) if cid.replace("CUST", "").isdigit() else 42)
            simulated_balance = int(monthly_income * np.random.uniform(0.35, 1.8))

            self.customers[cid] = {
                "customer_id": cid,
                "age": int(row["age"]) if pd.notnull(row["age"]) else 30,
                "occupation": str(row["occupation_type"]).replace("_", " ").title() if pd.notnull(row["occupation_type"]) else "Salaried",
                "annual_income": int(ann_inc),
                "monthly_income": monthly_income,
                "city_tier": str(row["city_tier"]).title() if pd.notnull(row["city_tier"]) else "Tier 2",
                "existing_loan_count": loan_cnt,
                "existing_loan_types": loan_types,
                "residence_type": str(row["residence_type"]).title() if pd.notnull(row["residence_type"]) else "Owned",
                "relationship_years": round(tenure, 1),
                "bureau_score": bureau,
                "bureau_grade": bureau_grade,
                "kyc_status": kyc_status,
                "kyc_status_hi": kyc_status_hi,
                "kyc_status_gu": kyc_status_gu,
                "kyc_verified": kyc_verified,
                "estimated_balance": simulated_balance,
            }

    def get_customer_ids(self, limit: int = 25) -> list:
        return list(self.customers.keys())[:limit]

    def get_profile(self, customer_id: str) -> dict:
        if customer_id in self.customers:
            return self.customers[customer_id]
        # Default fallback customer
        return {
            "customer_id": customer_id,
            "age": 32,
            "occupation": "Salaried",
            "annual_income": 600000,
            "monthly_income": 50000,
            "city_tier": "Metro",
            "existing_loan_count": 0,
            "existing_loan_types": [],
            "residence_type": "Owned",
            "relationship_years": 2.5,
            "bureau_score": 720,
            "bureau_grade": "Good / अच्छा / સારું",
            "kyc_status": "Fully Verified & Active (Aadhaar & PAN Linked)",
            "kyc_status_hi": "पूर्णतः सत्यापित और सक्रिय (आधार और पैन लिंक हैं)",
            "kyc_status_gu": "સંપૂર્ણ ચકાસાયેલ અને સક્રિય (આધાર અને પાન લિંક છે)",
            "kyc_verified": True,
            "estimated_balance": 48500,
        }

    def get_recent_transactions(self, customer_id: str, limit: int = 3) -> list:
        if customer_id in self.tx_cache:
            return self.tx_cache[customer_id][:limit]

        txs = []
        if os.path.exists(TRANSACTION_FILE):
            try:
                # Read chunks efficiently
                for chunk in pd.read_csv(TRANSACTION_FILE, chunksize=40000):
                    matches = chunk[chunk["customer_id"] == customer_id]
                    if not matches.empty:
                        for _, row in matches.iterrows():
                            txs.append({
                                "date": str(row["date"]),
                                "amount": abs(float(row["amount"])),
                                "direction": str(row["direction"]),
                                "channel": str(row["channel"]),
                                "narration": str(row["narration_raw"]),
                                "category": str(row["true_category"]).title(),
                            })
                    if len(txs) >= 15:
                        break
            except Exception as e:
                print(f"Error reading transaction ledger: {e}")

        # If customer found in transactions
        if txs:
            # Sort newest first
            txs.sort(key=lambda x: x["date"], reverse=True)
            self.tx_cache[customer_id] = txs
            return txs[:limit]

        # Simulated realistic default transactions
        profile = self.get_profile(customer_id)
        m_inc = profile.get("monthly_income", 45000)
        default_txs = [
            {"date": "2025-06-28", "amount": m_inc, "direction": "credit", "channel": "NEFT", "narration": "Monthly Salary Credit", "category": "Salary"},
            {"date": "2025-06-22", "amount": 1850.0, "direction": "debit", "channel": "UPI", "narration": "UPI/Grocery Mart", "category": "Shopping"},
            {"date": "2025-06-15", "amount": 850.0, "direction": "debit", "channel": "UPI", "narration": "UPI/Utility Electricity Bill", "category": "Bills"},
        ]
        self.tx_cache[customer_id] = default_txs
        return default_txs[:limit]
