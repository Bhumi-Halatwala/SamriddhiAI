"""app/shared/data_loader.py"""
from pathlib import Path
import pandas as pd
import joblib

PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")


def _cache():
    try:
        import streamlit as st
        return st.cache_data
    except ImportError:
        def passthrough(fn):
            return fn
        return passthrough


@_cache()
def load_model_input():
    return pd.read_parquet(PROCESSED / "model_input.parquet")


@_cache()
def load_stress():
    return pd.read_parquet(PROCESSED / "stress_features.parquet")


@_cache()
def load_anomaly():
    return pd.read_parquet(PROCESSED / "anomaly_features.parquet")


@_cache()
def load_life_stage():
    return pd.read_parquet(PROCESSED / "life_stage_features.parquet")


@_cache()
def load_profile():
    return pd.read_parquet(PROCESSED / "profile_features.parquet")


@_cache()
def load_txn_snapshot():
    return pd.read_parquet(PROCESSED / "txn_snapshot.parquet")


@_cache()
def load_behavior():
    return pd.read_parquet(PROCESSED / "behavior_features.parquet")


@_cache()
def load_recommender():
    return joblib.load(MODELS_DIR / "recommender.pkl")


def get_row(df, customer_id):
    sub = df[df["customer_id"] == customer_id]
    return sub.iloc[0] if len(sub) > 0 else None
