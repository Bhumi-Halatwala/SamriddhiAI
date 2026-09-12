# SamriddhiAI — AI-Powered Hyper-Personalized Banking for Bharat

> A prototype banking intelligence system that personalizes financial products for
> Tier 2/3 India, detects early signs of financial stress, and speaks to customers
> in their own language.

**Team:** Avengers · **Event:** HackOut 2026 · **Track:** Fintech / AI for Bharat

---

## Table of Contents

1. [The problem](#the-problem)
2. [Our solution](#our-solution)
3. [System architecture](#system-architecture)
4. [What we built](#what-we-built)
5. [Dataset](#dataset)
6. [Machine learning approach](#machine-learning-approach)
7. [Ethical safeguards & compliance](#ethical-safeguards--compliance)
8. [Known limitations](#known-limitations)
9. [Project structure](#project-structure)
10. [How to run](#how-to-run)
11. [Team](#team)
12. [Acknowledgements](#acknowledgements)

---

## The problem

Indian banks have built world-class digital infrastructure — UPI, video KYC, mobile
banking — yet a large share of customers in Tier 2/3/4 towns and rural India still
find banking apps confusing, generic, and disconnected from their actual financial
lives. Three specific failures:

1. **One-size-fits-all offers.** A 25-year-old renter in Indore and a 50-year-old
   business owner in Coimbatore get the same "pre-approved personal loan" pop-up.
2. **Language and literacy barriers.** First-time digital users drop off during
   onboarding and loan journeys because the interfaces assume English fluency and
   prior banking knowledge.
3. **Punitive rather than supportive responses to stress.** Banks tend to notice
   financial trouble only when a payment fails, and respond with default flags
   instead of empathy.

Meanwhile, banks sit on rich transactional and behavioural data that is barely
used for genuine personalization.

---

## Our solution

SamriddhiAI is a **prototype** that reads each customer's transaction history,
app engagement, and life-stage signals, then:

- **Recommends** the right banking product (loan, insurance, investment, credit
  card) at the right moment — with human-readable reasoning.
- **Detects early warning signals** of financial stress and behavioural anomaly,
  and triggers supportive rather than punitive interventions.
- **Speaks to customers** through a multilingual chatbot that works over text
  *and* voice, in English, Hindi, Tamil, Marathi, and Bengali.

We built two complementary interfaces:

- A **bank admin panel** for relationship managers: full customer snapshot,
  next-best-action, and risk alerts — all in plain language.
- A **customer app** for end users: spending patterns, personalized suggestions,
  and conversational banking in their preferred language.

The system enforces a **strict ethical guardrail**: customers showing signs of
financial stress are *never* offered new loans. Instead, they are routed to
advisory, EMI restructuring, or savings support.

---

## System architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              DATA LAYER                     │
                    │  Transactions · Behaviour · Profile · Labels│
                    └──────────────────────┬──────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │        FEATURE ENGINEERING PIPELINE          │
                    │  178 features per customer across:            │
                    │   • Profile (age, income, occupation, tier)   │
                    │   • Transactions (7-month rolling aggregates) │
                    │   • Behaviour (logins, calculator, sessions)  │
                    │   • Life stage (rule-based segmentation)      │
                    │   • Stress score (composite z-score)          │
                    └──────────────────────┬──────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │              AI ENGINE                       │
                    │                                              │
                    │  ┌──────────────┐  ┌──────────────────────┐  │
                    │  │ Narration    │  │  Recommender         │  │
                    │  │ Classifier   │  │  • Head A: propensity│  │
                    │  │ (LightGBM)   │  │  • Head B: product   │  │
                    │  └──────────────┘  └──────────────────────┘  │
                    │                                              │
                    │  ┌──────────────┐  ┌──────────────────────┐  │
                    │  │ Anomaly      │  │  Chatbot             │  │
                    │  │ Detector     │  │  • Intent routing    │  │
                    │  │ (Isolation   │  │  • 5 languages       │  │
                    │  │  Forest +    │  │  • Text + Voice      │  │
                    │  │  rule flags) │  │  • Guardrail-aware   │  │
                    │  └──────────────┘  └──────────────────────┘  │
                    └──────────────────────┬──────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │            GUARDRAIL LAYER                   │
                    │  • Suppress loan offers for stressed users    │
                    │  • Empathetic interventions over flags        │
                    │  • Bias audit across segments                 │
                    └──────────────────────┬──────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                              │
                    ▼                                              ▼
        ┌──────────────────────┐                    ┌──────────────────────┐
        │  BANK ADMIN PANEL    │                    │   CUSTOMER APP       │
        │  (port 8501)         │                    │   (port 8502)        │
        │  • Snapshot          │                    │  • My Account        │
        │  • Next best action  │                    │  • My Money          │
        │  • Risk & alerts     │                    │  • For You           │
        └──────────────────────┘                    └──────────────────────┘
```

---

## What we built

### 1. Narration parser (NLP layer)

Bank transaction narrations arrive as messy strings like
`NEFT/416618/SAL/BHARATRETA` or `ECS-RETURN-INSUFFICIENT FUNDS`. We trained a
TF-IDF + LightGBM classifier that maps any narration to one of seven categories:
`salary`, `business_income`, `emi`, `rent`, `discretionary`, `bounce`,
`suspicious`.

- **On the templated training data:** F1 macro = 1.00
- **On 40 hand-crafted messy narrations (adversarial test):** F1 macro = 0.43
- **With rule-based fallback:** F1 macro = 0.84

The adversarial test is deliberate — we treat the narration parser as a
production component that will encounter distribution shift, and we designed a
hybrid (ML + rules) system accordingly.

### 2. Recommendation engine

Two-head LightGBM model:

- **Head A — propensity:** P(customer applies for any loan) — test AUC = 0.61
- **Head B — product:** P(which product | they apply) — test macro-F1 = 0.99

The full 178-feature model achieves AUC 0.609. A **5-feature explainable
variant** achieves AUC 0.605 — essentially identical. We ship both, and prefer
the parsimonious version for regulatory clarity.

### 3. Anomaly & stress detection

Combination of:

- **Isolation Forest** on per-customer z-score summaries across 18 months
- **Rule-based hard flags:** bounce spikes, income gaps, savings drops, window-dressing patterns
- **Composite stress score** combining payment difficulty, savings erosion, discretionary behaviour change, and salary momentum

Performance on validation:

- **F1 = 0.48** on 150 known `window_dressing` customers (unsupervised, no labels used)
- **Recall = 0.76, lift = 26.4×** on 100 synthetically injected anomalies
- **Three-tier alerts** (Normal / Unusual / Investigate) with balanced distribution

### 4. Conversational AI (chatbot)

- Five languages: English, हिंदी, தமிழ், मराठी, বাংলা
- **Text + speech:** users can type or speak; the bot can respond with text or voice
- **Deterministic intent routing** (rule-based for demo reliability)
- **Guardrail-aware:** if the customer is stressed, the bot refuses to discuss
  new loans and redirects to advisory

### 5. Two-interface system

| Interface | Audience | Tabs |
|---|---|---|
| Bank admin panel | Relationship managers | Snapshot · Next best action · Risk & alerts |
| Customer app | End customers | My Account · My Money · For You |

All panels are built in Streamlit and run locally with no cloud dependencies.

---

## Dataset

We use a synthetic Indian bank dataset with four files:

| File | Rows | Purpose |
|---|---|---|
| `customer_master.csv` | 5,000 | Demographics, income, existing loans, bureau score |
| `labels.csv` | 5,000 | Loan application and conversion outcomes |
| `transaction_ledger.csv` | 1,033,406 | 18 months of transactions (Jan 2024 – Jun 2025) |
| `behavioral_log.csv` | 489,663 | 6 months of app activity (Jan – Jun 2025) |

**Label balance:** 39.8% applied, 31.1% converted — no SMOTE needed.

**Data quality checks we performed:**
- Referential integrity: 5,000/5,000 customers in every table, zero orphans
- Leakage diagnostic: behavioural gap between applied/non-applied customers is
  flat across all 18 months — no leakage from ledger to label
- Time windows: features snapshotted as of Jan 2025; June 2025 excluded from
  training (partial month)

---

## Machine learning approach

### Feature engineering (178 features per customer)

- **Profile (23):** age bands, income (raw + log), existing-loan-type flags, city-tier one-hots, bureau bands
- **Transaction (97):** 7-month rolling aggregates of 14 base features × 6 aggregations (mean, std, last-3, first-3, momentum, slope), plus regularity and anomaly signals
- **Behaviour (48):** login counts, session stats, product-page views, calculator usage, engagement score, funnel stage
- **Life stage (12):** 8 rule-based segments + one-hots (young renter, young family, mortgage holder, debt-stressed, etc.)
- **Stress (17):** composite z-score with weighted signals, three-tier bucket, per-customer anomaly counts

### Models

| Model | Algorithm | Purpose | Key metric |
|---|---|---|---|
| Narration classifier | TF-IDF + LightGBM | Map raw narrations to categories | F1 = 0.84 (adversarial) |
| Recommendation Head A | LightGBM binary | Predict loan application | AUC = 0.61 |
| Recommendation Head B | LightGBM multiclass | Predict product choice | F1 = 0.99 |
| Anomaly detector | Isolation Forest | Flag unusual monthly patterns | Lift = 26.4× |
| Stress score | Weighted z-sum | Composite financial health | Correlates with bounce rate (48% vs 13%) |

### Explainability

- **SHAP values** on Head A show `engagement_score`, `total_events`, and
  transaction trends as top drivers
- **Rule-based reason codes** surface in both panels as plain-English explanations
  ("Receives regular monthly salary for 12+ months")
- **Model card** documents training data, evaluation, and known limitations

---

## Ethical safeguards & compliance

### Guardrails

- **No predatory nudging:** a hard-coded policy in the recommender *and* the
  chatbot suppresses all loan offers for customers classified as high-stress.
- **Empathetic interventions:** stressed customers receive advisory support, EMI
  restructuring options, and savings help — not default flags.
- **Reason-code transparency:** every recommendation is accompanied by
  human-readable reasoning; no black-box decisions.
- **Bias audit:** we check recommendation rate across city tier, occupation, and
  age band. The dataset shows no disparate impact (flat application rate across
  segments), which we document as a finding, not a virtue.

### Data privacy & regulatory alignment

- **DPDP Act (2023):** data minimization (only fields used are documented),
  purpose limitation, right to explanation.
- **RBI data localization:** the prototype runs locally with no cross-border
  data transfer; production would deploy in `ap-south-1` (Mumbai).
- **Consent:** both panels assume explicit customer consent for personalization;
  production would include a runtime consent toggle.
- **No PII in logs:** the ML pipeline operates on `customer_id` only; no names,
  phone numbers, or addresses are used anywhere.

### Documented limitations (not hidden)

- **No missed-EMI signal in the dataset.** We verified this and redistributed
  the stress score's weight to bounces. Production would add missed EMI with
  the original 0.30 weight.
- **No real fraud labels.** `window_dressing` provides one ground-truth signal
  (150 customers); the rest of the anomaly evaluation is on synthetic injections.
- **Bureau score correlation ≈ 0.** We kept the feature and documented its
  low importance; this is a dataset property, not a design choice.
- **Chatbot uses deterministic routing.** A live LLM can be swapped in but was
  not used in the demo to avoid hallucination risks.


---

## Known limitations

We would rather document these than hide them.

1. **Weak Head A signal (AUC 0.61).** The synthetic dataset shows little
   demographic signal in the labels. This is consistent with our leakage
   diagnostic and with the flat application rate across segments.
2. **Trivial narration parser on templated data.** Template-generated narrations
   are too clean; we built the adversarial test to expose the deployment gap and
   mitigate with rules.
3. **Anomaly detector F1 = 0.48 on window_dressing.** Respectable for an
   unsupervised method with no labels, but not production-grade.
4. **Chatbot is rule-based.** Sufficient for the demo; production would use an
   LLM grounded in the customer's live state, with a rules layer for safety.
5. **Tier 4 coverage absent.** Dataset only has metro/tier2/tier3. We treat
   tier2/tier3 as the "Bharat" proxy and note that tier4 requires additional
   data collection.

---

## Project structure

```
SamriddhiAI/
├── data/
│   ├── raw/                          # Original CSVs (not committed)
│   ├── interim/                      # Cleaned parquets (Phase 1)
│   └── processed/                    # Feature tables + model inputs
│       ├── profile_features.parquet
│       ├── txn_monthly.parquet
│       ├── txn_snapshot.parquet
│       ├── behavior_features.parquet
│       ├── life_stage_features.parquet
│       ├── stress_features.parquet
│       ├── anomaly_features.parquet
│       ├── model_input.parquet
│       └── labels_joined.parquet
│
├── src/
│   ├── data/                         # Loading, validation, EDA
│   │   ├── load.py
│   │   ├── eda.py
│   │   ├── check_setup.py
│   │   └── diagnose_label_window.py
│   ├── features/                     # Feature engineering
│   │   ├── profile_features.py
│   │   ├── txn_features.py
│   │   ├── behavior_features.py
│   │   ├── life_stage.py
│   │   ├── stress.py
│   │   └── build_model_input.py
│   ├── models/                       # Model training + inference
│   │   ├── narration_clf.py
│   │   ├── predict_narration.py
│   │   ├── adversarial_test.py
│   │   ├── recommender.py
│   │   └── anomaly.py
│   └── chatbot/                      # Conversational AI
│       ├── languages.py
│       ├── intents.py
│       ├── router.py
│       ├── demo.py
│       └── [speech.py]               # Text-to-speech & speech-to-text
│
├── app/                              # Streamlit front-ends
│   ├── bank_panel.py                 # Bank admin interface (port 8501)
│   ├── user_panel.py                 # Customer interface (port 8502)
│   └── shared/
│       ├── data_loader.py
│       └── formatting.py
│
├── models/                           # Saved model artifacts
│   ├── narration_clf.pkl
│   ├── recommender.pkl
│   └── anomaly.pkl
│
├── outputs/
│   ├── figures/                      # All plots
│   └── reports/                      # Phase-by-phase reports
│
├── tests/                            # Pytest test suite
│   ├── test_load.py
│   ├── test_narration.py
│   ├── test_features.py
│   ├── test_chatbot.py
│   └── [test_recommender.py]
│
├── docs/                             # Model card, ethics, compliance
│   ├── model_card.md
│   ├── ethics_dpdp.md
│   └── rbi_compliance.md
│
├── requirements.txt
└── README.md
```

---

## How to run

### Prerequisites

- Python 3.11
- Windows / macOS / Linux
- ~2 GB disk space for parquet artifacts

### Setup

```bash
# From the SamriddhiAI/ directory
py -3.11 -m pip install -r requirements.txt
```

### Data

Place the four raw CSVs in `data/raw/`:

```
data/raw/customer_master.csv
data/raw/transaction_ledger.csv
data/raw/behavioral_log.csv
data/raw/labels.csv
```

### Build the pipeline

Run these in order. Each is idempotent and can be safely re-run.

```bash
# Phase 1 — Load, validate, and save parquets
py -3.11 -m src.data.load
py -3.11 -m src.data.eda

# Phase 2 — Narration parser
py -3.11 -m src.models.narration_clf
py -3.11 -m src.models.adversarial_test

# Phase 3 — Feature engineering
py -3.11 -m src.features.profile_features
py -3.11 -m src.features.txn_features
py -3.11 -m src.features.behavior_features
py -3.11 -m src.features.life_stage
py -3.11 -m src.features.stress
py -3.11 -m src.features.build_model_input

# Phase 4 — Recommender
py -3.11 -m src.models.recommender

# Phase 5 — Anomaly detection
py -3.11 -m src.models.anomaly
```

### Run the demo apps

Open two terminals:

```bash
# Terminal 1 — Bank admin panel
py -3.11 -m streamlit run app/bank_panel.py --server.port 8501

# Terminal 2 — Customer app
py -3.11 -m streamlit run app/user_panel.py --server.port 8502
```

Then open:
- Bank panel: [http://localhost:8501](http://localhost:8501)
- Customer app: [http://localhost:8502](http://localhost:8502)

**Demo tip:** try `CUST000005` (healthy finances — will see product offers) and
`CUST000001` (financially stressed — will see supportive messaging and no loan
offers).

### Run tests

```bash
py -3.11 -m pytest tests/ -v
```

---

## Team

**Team Avengers** — HackOut 2026

| # | Name | Role | ID |
|---|---|---|---|
| 1 | Devanshi Vora | 202618013 |
| 2 | Khushboo Dharmani | 202618020 |
| 3 | Bhumi Halatwala | 202618038 |
| 4 | Vrinda Teli | 202618050 |

---

## Acknowledgements

- Dataset: https://www.kaggle.com/datasets/shuchismitamallick/loan-underwriting-and-customer-behavior-dataset?
- Built with: Python 3.11, pandas, scikit-learn, LightGBM, SHAP, Streamlit
- Special thanks to Dhirubhai Ambani University for this opportunity

---

## License

For academic / hackathon use only

---

*Built for Bharat. Powered by AI. Guided by ethics.*
