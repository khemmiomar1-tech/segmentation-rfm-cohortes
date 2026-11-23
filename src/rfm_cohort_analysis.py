"""
RFM segmentation and cohort retention analysis.

Steps:
  1. Clean the raw transactions (duplicates, invalid amounts, outliers).
  2. Compute Recency / Frequency / Monetary per customer and score each
     dimension in quintiles (1-5). Segments are assigned from the R and F
     scores with an ordered rule set (Champions, Loyal, At Risk, Lost...).
  3. Simplified CLV per segment: average basket x orders per year x 2-year
     horizon, averaged per segment.
  4. Cohort analysis: customers grouped by first-purchase month; retention
     matrix = share of each cohort still active N months later.

Usage:
    python src/rfm_cohort_analysis.py
Inputs:
    data/sample/transactions.csv
Outputs (output/):
    rfm_segments.csv     one row per customer: R/F/M values, scores, segment
    segment_summary.csv  customers, revenue share, avg RFM, simplified CLV,
                         and one recommended marketing action per segment
    cohort_retention.csv retention matrix in % (rows = cohort month,
                         columns = months since first purchase)
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "sample" / "transactions.csv"
OUT = BASE / "output"

# One recommended marketing action per segment (used in the summary table).
SEGMENT_ACTIONS = {
    "Champions": "Reward and involve: early access, referral programme, VIP offers.",
    "Loyal": "Keep engaged: loyalty programme, personalised cross-sell.",
    "Potential Loyalists": "Convert: targeted offers and reminders to increase frequency.",
    "New Customers": "Onboard: welcome journey and second-purchase incentive.",
    "Promising": "Nurture: product discovery content and small incentives.",
    "At Risk": "Win back: reactivation campaign with a time-limited discount.",
    "Hibernating": "Re-engage cheaply: low-cost email reminder, survey.",
    "Lost": "Low priority: one last-chance offer, then let go.",
    "Others": "Monitor: standard campaigns.",
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, non-positive amounts and extreme amount outliers."""
    n0 = len(df)
    df = df.drop_duplicates()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date", "customer_id"])
    df = df[df["amount"] > 0]
    # Outliers: keep amounts below the 99.9th percentile (data-entry errors).
    cap = df["amount"].quantile(0.999)
    df = df[df["amount"] <= cap]
    print(f"Cleaning: {n0:,} raw rows -> {len(df):,} valid rows "
          f"({n0 - len(df)} removed)")
    return df


def quintile_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Score R/F/M in quintiles 1-5. rank(method='first') makes qcut robust
    to the many tied values that are typical for Frequency."""
    rfm["R_score"] = pd.qcut(rfm["recency"].rank(method="first"), 5,
                             labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5,
                             labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5,
                             labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_score"] = (rfm["R_score"].astype(str) + rfm["F_score"].astype(str)
                        + rfm["M_score"].astype(str))
    return rfm


def assign_segment(r: int, f: int) -> str:
    """Ordered rules on the R and F scores (first match wins)."""
    if r >= 4 and f >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r >= 4 and f >= 2:
        return "Potential Loyalists"
    if r >= 4 and f == 1:
        return "New Customers"
    if r == 3:
        return "Promising"
    if r <= 2 and f >= 2:
        return "At Risk"
    if r == 2 and f == 1:
        return "Hibernating"
    if r == 1 and f == 1:
        return "Lost"
    return "Others"


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["order_date"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("customer_id")
        .agg(
            recency=("order_date", lambda d: (snapshot - d.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("amount", "sum"),
            first_purchase=("order_date", "min"),
            last_purchase=("order_date", "max"),
        )
        .reset_index()
    )
    rfm["monetary"] = rfm["monetary"].round(2)
    rfm = quintile_scores(rfm)
    rfm["segment"] = [assign_segment(r, f)
                      for r, f in zip(rfm["R_score"], rfm["F_score"])]

    # Simplified CLV: avg basket x orders/year x 2-year horizon.
    obs_years = (df["order_date"].max() - df["order_date"].min()).days / 365.25
    rfm["avg_basket"] = rfm["monetary"] / rfm["frequency"]
    rfm["clv_simple"] = (rfm["avg_basket"] * (rfm["frequency"] / obs_years) * 2).round(2)
    return rfm


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    total_rev = rfm["monetary"].sum()
    summary = (
        rfm.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            revenue=("monetary", "sum"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            avg_clv_simple=("clv_simple", "mean"),
        )
        .reset_index()
    )
    summary["customers_pct"] = (summary["customers"] / len(rfm) * 100).round(1)
    summary["revenue_pct"] = (summary["revenue"] / total_rev * 100).round(1)
    summary = summary.round({"revenue": 2, "avg_recency": 1, "avg_frequency": 2,
                             "avg_monetary": 2, "avg_clv_simple": 2})
    summary["recommended_action"] = summary["segment"].map(SEGMENT_ACTIONS)
    return summary.sort_values("revenue", ascending=False).reset_index(drop=True)


def cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    """Retention matrix: % of each first-purchase cohort active N months later."""
    df = df.copy()
    df["order_month"] = df["order_date"].dt.to_period("M")
    first = df.groupby("customer_id")["order_month"].min().rename("cohort_month")
    df = df.join(first, on="customer_id")
    df["cohort_index"] = ((df["order_month"].dt.year - df["cohort_month"].dt.year) * 12
                          + (df["order_month"].dt.month - df["cohort_month"].dt.month))

    active = (df.groupby(["cohort_month", "cohort_index"])["customer_id"]
              .nunique().unstack(fill_value=0))
    cohort_sizes = first.value_counts().sort_index()
    retention = active.div(cohort_sizes, axis=0).mul(100).round(1)
    retention.insert(0, "cohort_size", cohort_sizes)
    retention.index = retention.index.astype(str)
    retention.columns = [str(c) for c in retention.columns]
    return retention


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = clean(pd.read_csv(RAW))

    rfm = build_rfm(df)
    rfm_out = rfm[["customer_id", "recency", "frequency", "monetary",
                   "R_score", "F_score", "M_score", "RFM_score", "segment",
                   "clv_simple"]]
    rfm_out.to_csv(OUT / "rfm_segments.csv", index=False)

    summary = segment_summary(rfm)
    summary.to_csv(OUT / "segment_summary.csv", index=False)

    retention = cohort_retention(df)
    retention.to_csv(OUT / "cohort_retention.csv", index_label="cohort_month")

    print(f"Customers segmented : {len(rfm):,}")
    print(f"Segments found      : {rfm['segment'].nunique()}")
    print(summary[["segment", "customers", "customers_pct",
                   "revenue_pct", "avg_clv_simple"]].to_string(index=False))
    print(f"Outputs written to {OUT}/")


if __name__ == "__main__":
    main()
