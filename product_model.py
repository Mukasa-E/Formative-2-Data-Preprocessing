"""
Formative 2 - Product Recommendation Model
==========================================
Pipeline: load -> clean -> merge -> EDA -> feature engineering -> train -> evaluate -> save.

Run from the repo root:
    python product_model.py

Expected layout:
    data/customer_social_profiles.xlsx
    data/customer_transactions.xlsx
Outputs:
    data/merged_dataset.csv            (merged + engineered dataset)
    models/product_model.joblib        (trained model + feature list, used by the CLI app)
    plots/*.png                        (EDA and evaluation figures)
    models/product_metrics.json        (accuracy, F1, loss for every model)

The CLI app only needs:
    from product_model import predict_product
    predict_product(customer_id)  ->  (product_name, confidence)
"""

import json
import os
import warnings

import joblib
import sys

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # scripts: no display needed, figures are saved to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # keep the script usable without xgboost
    HAS_XGB = False

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
DATA_DIR = os.environ.get("DATA_DIR", "data")
MODEL_DIR = "models"
PLOT_DIR = "plots"
for d in (MODEL_DIR, PLOT_DIR):
    os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
def load_data():
    social = pd.read_excel(os.path.join(DATA_DIR, "customer_social_profiles.xlsx"))
    trans = pd.read_excel(os.path.join(DATA_DIR, "customer_transactions.xlsx"))
    return social, trans


# ----------------------------------------------------------------------------
# 2. CLEAN
# ----------------------------------------------------------------------------
def clean_data(social, trans, verbose=True):
    log = {}

    # --- social profiles ---
    log["social_raw_shape"] = social.shape
    log["social_exact_duplicates"] = int(social.duplicated().sum())
    social = social.drop_duplicates().copy()

    # The two files use different ID formats: "A178" (social) vs 178 (transactions).
    # Strip the "A" prefix and cast to int so both keys share one type and meaning.
    social["customer_id"] = (
        social["customer_id_new"].astype(str).str.strip().str.upper()
        .str.replace("A", "", regex=False).astype(int)
    )
    social = social.drop(columns="customer_id_new")
    for c in ["social_media_platform", "review_sentiment"]:
        social[c] = social[c].astype(str).str.strip().str.title()

    # --- transactions ---
    log["trans_raw_shape"] = trans.shape
    log["trans_exact_duplicates"] = int(trans.duplicated().sum())
    log["trans_duplicate_transaction_ids"] = int(trans["transaction_id"].duplicated().sum())
    trans = trans.drop_duplicates().copy()
    trans = trans.rename(columns={"customer_id_legacy": "customer_id"})
    trans["customer_id"] = trans["customer_id"].astype(int)
    trans["purchase_date"] = pd.to_datetime(trans["purchase_date"])
    trans["product_category"] = trans["product_category"].astype(str).str.strip().str.title()

    # Missing ratings: keep the row (a purchase is still a purchase) and impute with the
    # median (robust to outliers). Add a flag so the model/report knows it was imputed.
    log["trans_missing_rating"] = int(trans["customer_rating"].isna().sum())
    trans["rating_was_missing"] = trans["customer_rating"].isna().astype(int)
    trans["customer_rating"] = trans["customer_rating"].fillna(trans["customer_rating"].median())

    log["social_clean_shape"] = social.shape
    log["trans_clean_shape"] = trans.shape
    log["social_unique_customers"] = int(social["customer_id"].nunique())
    log["trans_unique_customers"] = int(trans["customer_id"].nunique())
    if verbose:
        print("--- CLEANING LOG ---")
        for k, v in log.items():
            print(f"{k:35s}: {v}")
    return social, trans, log


# ----------------------------------------------------------------------------
# 3. MERGE
# ----------------------------------------------------------------------------
def aggregate_social(social):
    """One row per customer. Joining the raw social table directly would be a
    many-to-many join (a customer appears several times in BOTH files) and would
    multiply transaction rows, inflating the dataset with fake records."""
    mode = lambda s: s.mode().iloc[0]
    agg = social.groupby("customer_id").agg(
        avg_engagement_score=("engagement_score", "mean"),
        avg_purchase_interest=("purchase_interest_score", "mean"),
        n_social_records=("engagement_score", "size"),
        main_platform=("social_media_platform", mode),
        main_sentiment=("review_sentiment", mode),
    ).reset_index()
    return agg


def merge_data(social, trans, verbose=True):
    social_agg = aggregate_social(social)

    # LEFT join on transactions: the transaction table holds the prediction target
    # (product_category), so we keep every transaction and attach social features
    # where they exist. Customers without a social profile get a flag + neutral fill.
    merged = trans.merge(social_agg, on="customer_id", how="left", validate="m:1")
    merged["has_social_profile"] = merged["avg_engagement_score"].notna().astype(int)

    checks = {
        "rows_before_merge (transactions)": len(trans),
        "rows_after_merge": len(merged),
        "row_count_preserved": len(trans) == len(merged),
        "customers_with_social_profile": int(merged.loc[merged.has_social_profile == 1, "customer_id"].nunique()),
        "customers_without_social_profile": int(merged.loc[merged.has_social_profile == 0, "customer_id"].nunique()),
        "transactions_without_social_data": int((merged.has_social_profile == 0).sum()),
        "duplicate_transaction_ids_after_merge": int(merged["transaction_id"].duplicated().sum()),
        "inner_join_rows_for_comparison": int(len(trans.merge(social_agg, on="customer_id", how="inner"))),
    }
    # Fill missing social features
    for c in ["avg_engagement_score", "avg_purchase_interest"]:
        merged[c] = merged[c].fillna(merged[c].median())
    merged["n_social_records"] = merged["n_social_records"].fillna(0)
    merged["main_platform"] = merged["main_platform"].fillna("Unknown")
    merged["main_sentiment"] = merged["main_sentiment"].fillna("Unknown")

    if verbose:
        print("\n--- POST-MERGE CHECKS ---")
        for k, v in checks.items():
            print(f"{k:42s}: {v}")
        print("Nulls remaining:", int(merged.isna().sum().sum()))
    return merged, checks


# ----------------------------------------------------------------------------
# 4. EDA
# ----------------------------------------------------------------------------
def run_eda(df):
    print("\n--- SUMMARY STATISTICS ---")
    print(df.describe().round(2).to_string())
    print("\n--- VARIABLE TYPES ---")
    print(df.dtypes.to_string())

    # Plot 1: target distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    order = df["product_category"].value_counts().index
    sns.countplot(data=df, x="product_category", order=order, palette="viridis", ax=ax)
    ax.set_title("Distribution of Product Categories (target)")
    ax.set_xlabel("Product category"); ax.set_ylabel("Number of transactions")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(f"{PLOT_DIR}/eda_1_target_distribution.png", dpi=150); plt.close(fig)

    # Plot 2: purchase amount distribution + outliers (box plot by category)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["purchase_amount"], bins=20, kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title("Purchase Amount Distribution")
    sns.boxplot(data=df, x="product_category", y="purchase_amount", order=order, ax=axes[1], palette="viridis")
    axes[1].set_title("Purchase Amount by Category (outlier check)")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(f"{PLOT_DIR}/eda_2_purchase_amount.png", dpi=150); plt.close(fig)

    # Plot 3: correlation heatmap
    num = ["purchase_amount", "customer_rating", "avg_engagement_score",
           "avg_purchase_interest", "n_social_records", "has_social_profile"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (numeric features)")
    fig.tight_layout(); fig.savefig(f"{PLOT_DIR}/eda_3_correlation_heatmap.png", dpi=150); plt.close(fig)

    # Plot 4: rating and social features by category
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=df, x="product_category", y="customer_rating", order=order, ax=axes[0], palette="viridis")
    axes[0].set_title("Customer Rating by Category"); axes[0].tick_params(axis="x", rotation=20)
    sns.boxplot(data=df[df.has_social_profile == 1], x="product_category", y="avg_purchase_interest",
                order=order, ax=axes[1], palette="viridis")
    axes[1].set_title("Social Purchase-Interest Score by Category"); axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(f"{PLOT_DIR}/eda_4_rating_interest_by_category.png", dpi=150); plt.close(fig)

    # Plot 5: sentiment / platform vs category
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    pd.crosstab(df["main_platform"], df["product_category"]).plot(kind="bar", stacked=True, ax=axes[0], colormap="viridis")
    axes[0].set_title("Product Category by Main Social Platform"); axes[0].tick_params(axis="x", rotation=20)
    pd.crosstab(df["main_sentiment"], df["product_category"]).plot(kind="bar", stacked=True, ax=axes[1], colormap="viridis")
    axes[1].set_title("Product Category by Review Sentiment"); axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(f"{PLOT_DIR}/eda_5_platform_sentiment.png", dpi=150); plt.close(fig)

    # Numeric outlier count (IQR rule) for the report
    q1, q3 = df["purchase_amount"].quantile([.25, .75]); iqr = q3 - q1
    n_out = int(((df.purchase_amount < q1 - 1.5 * iqr) | (df.purchase_amount > q3 + 1.5 * iqr)).sum())
    print(f"\nPurchase-amount outliers (IQR rule): {n_out}")

    # Statistical tests: does any feature actually relate to the product category?
    from scipy import stats
    print("\n--- FEATURE vs PRODUCT CATEGORY (significance tests, alpha = 0.05) ---")
    tests = {}
    for col in ["purchase_amount", "customer_rating", "avg_engagement_score", "avg_purchase_interest"]:
        groups = [g[col].values for _, g in df.groupby("product_category")]
        f, p = stats.f_oneway(*groups)
        tests[col] = round(float(p), 4)
        print(f"ANOVA  {col:24s} F={f:6.3f}  p={p:.4f}")
    for col in ["main_platform", "main_sentiment"]:
        chi, p, _, _ = stats.chi2_contingency(pd.crosstab(df[col], df["product_category"]))
        tests[col] = round(float(p), 4)
        print(f"Chi2   {col:24s} chi2={chi:6.3f}  p={p:.4f}")
    return {"purchase_amount_outliers_iqr": n_out, "p_values": tests}


# ----------------------------------------------------------------------------
# 5. FEATURE ENGINEERING
# ----------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy().sort_values(["purchase_date", "transaction_id"]).reset_index(drop=True)

    # Date features
    df["purchase_month"] = df["purchase_date"].dt.month
    df["purchase_dayofweek"] = df["purchase_date"].dt.dayofweek
    df["is_weekend"] = (df["purchase_dayofweek"] >= 5).astype(int)

    # Customer-level behaviour (never uses the target column -> no label leakage)
    g = df.groupby("customer_id")
    df["customer_n_purchases"] = g["transaction_id"].transform("count")
    df["customer_avg_amount"] = g["purchase_amount"].transform("mean")
    df["customer_avg_rating"] = g["customer_rating"].transform("mean")
    df["amount_vs_customer_avg"] = df["purchase_amount"] - df["customer_avg_amount"]

    # Interaction between social signals
    df["engagement_x_interest"] = df["avg_engagement_score"] * df["avg_purchase_interest"]

    # Bucketed amount
    df["amount_bucket"] = pd.qcut(df["purchase_amount"], q=4, labels=["low", "mid", "high", "premium"])

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=["main_platform", "main_sentiment", "amount_bucket"], drop_first=False, dtype=int)
    return df


def get_feature_columns(df):
    drop = {"customer_id", "transaction_id", "purchase_date", "product_category", "product_label"}
    return [c for c in df.columns if c not in drop]


# ----------------------------------------------------------------------------
# 6. TRAIN + EVALUATE
# ----------------------------------------------------------------------------
def train_and_evaluate(df):
    feats = get_feature_columns(df)
    le = LabelEncoder()
    y = le.fit_transform(df["product_category"])
    X = df[feats].astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_tr)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=0.5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1),
    }
    if HAS_XGB:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=2.0, eval_metric="mlogloss",
            random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = {}, {}
    print("\n--- MODEL COMPARISON ---")
    for name, model in candidates.items():
        use_scaled = name == "Logistic Regression"
        Xtr = scaler.transform(X_tr) if use_scaled else X_tr
        Xte = scaler.transform(X_te) if use_scaled else X_te
        model.fit(Xtr, y_tr)
        proba = model.predict_proba(Xte)
        pred = proba.argmax(axis=1)
        cv_acc = cross_val_score(model, scaler.transform(X) if use_scaled else X, y, cv=cv, scoring="accuracy").mean()
        results[name] = {
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "f1_weighted": round(f1_score(y_te, pred, average="weighted"), 4),
            "f1_macro": round(f1_score(y_te, pred, average="macro"), 4),
            "log_loss": round(log_loss(y_te, proba, labels=range(len(le.classes_))), 4),
            "cv_accuracy_5fold": round(cv_acc, 4),
        }
        fitted[name] = (model, use_scaled)
        print(f"{name:20s} {results[name]}")

    baseline = max(np.bincount(y_te)) / len(y_te)
    uniform_loss = float(np.log(len(le.classes_)))
    print(f"\nMajority-class baseline accuracy: {baseline:.4f}")
    print(f"Random-guess log loss (ln {len(le.classes_)}): {uniform_loss:.4f}")

    # Pick the best model by weighted F1 (ties -> lower log loss)
    best_name = sorted(results, key=lambda n: (-results[n]["f1_weighted"], results[n]["log_loss"]))[0]
    best_model, best_scaled = fitted[best_name]
    print(f"\nSelected model: {best_name}")

    Xte_best = scaler.transform(X_te) if best_scaled else X_te
    pred = best_model.predict(Xte_best)
    print("\n--- CLASSIFICATION REPORT (test set) ---")
    print(classification_report(y_te, pred, target_names=le.classes_, zero_division=0))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(confusion_matrix(y_te, pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f"Confusion Matrix - {best_name}"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(f"{PLOT_DIR}/model_confusion_matrix.png", dpi=150); plt.close(fig)

    # Model comparison bar chart
    comp = pd.DataFrame(results).T[["accuracy", "f1_weighted", "cv_accuracy_5fold"]]
    ax = comp.plot(kind="bar", figsize=(8, 4.5), colormap="viridis")
    ax.axhline(baseline, color="red", ls="--", label=f"majority baseline ({baseline:.2f})")
    ax.set_title("Model Comparison"); ax.set_ylabel("Score"); ax.legend(); plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/model_comparison.png", dpi=150); plt.close()

    # Feature importance (tree models)
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=feats).sort_values().tail(12)
        fig, ax = plt.subplots(figsize=(7, 5))
        imp.plot(kind="barh", color="teal", ax=ax)
        ax.set_title(f"Top Feature Importances - {best_name}")
        fig.tight_layout(); fig.savefig(f"{PLOT_DIR}/model_feature_importance.png", dpi=150); plt.close(fig)

    # Save everything the CLI needs
    artifact = {
        "model": best_model, "model_name": best_name, "scaler": scaler if best_scaled else None,
        "label_encoder": le, "features": feats,
    }
    joblib.dump(artifact, f"{MODEL_DIR}/product_model.joblib")
    metrics = {"models": results, "selected_model": best_name,
               "majority_baseline_accuracy": round(float(baseline), 4),
               "random_guess_log_loss": round(uniform_loss, 4),
               "n_train": int(len(X_tr)), "n_test": int(len(X_te)), "n_features": len(feats)}
    with open(f"{MODEL_DIR}/product_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return artifact, metrics


# ----------------------------------------------------------------------------
# 7. INFERENCE (used by the CLI app)
# ----------------------------------------------------------------------------
_CACHE = {}


def predict_product(customer_id, merged_path=None, model_path=None):
    """Predict the product category for a customer using their latest transaction
    profile from merged_dataset.csv. Returns (product_name, confidence 0-1)."""
    merged_path = merged_path or os.path.join(DATA_DIR, "merged_dataset.csv")
    model_path = model_path or os.path.join(MODEL_DIR, "product_model.joblib")
    if "art" not in _CACHE:
        _CACHE["art"] = joblib.load(model_path)
        _CACHE["df"] = pd.read_csv(merged_path, parse_dates=["purchase_date"])
    art, df = _CACHE["art"], _CACHE["df"]
    rows = df[df["customer_id"] == int(customer_id)]
    if rows.empty:
        raise ValueError(f"customer_id {customer_id} not found in merged dataset")
    row = rows.sort_values("purchase_date").iloc[[-1]][art["features"]].astype(float)
    X = art["scaler"].transform(row) if art["scaler"] is not None else row
    proba = art["model"].predict_proba(X)[0]
    idx = int(proba.argmax())
    return art["label_encoder"].classes_[idx], float(proba[idx])


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    social, trans = load_data()
    social, trans, _ = clean_data(social, trans)
    merged, _ = merge_data(social, trans)
    run_eda(merged)
    final = engineer_features(merged)
    final.to_csv(os.path.join(DATA_DIR, "merged_dataset.csv"), index=False)
    print(f"\nSaved merged dataset: {final.shape} -> {DATA_DIR}/merged_dataset.csv")
    train_and_evaluate(final)

    # quick demo of the function the CLI will call
    cid = int(final["customer_id"].iloc[0])
    print(f"\nDemo prediction for customer {cid}: {predict_product(cid)}")


if __name__ == "__main__":
    main()
