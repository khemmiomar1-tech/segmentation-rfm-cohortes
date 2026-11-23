# Customer Segmentation (RFM) & Cohort Analysis

An e-commerce customer analysis in **Python (pandas)**: transactional data is
cleaned, every customer is scored with the **RFM** method (Recency, Frequency,
Monetary), assigned to a behavioural segment, valued with a **simplified
CLV**, and tracked over time with a **cohort retention matrix** — the outputs
feed a Power BI report and one recommended marketing action per segment.

> The dataset in `data/sample/` is **100% synthetic** (fictional customers
> and amounts, generated with a fixed seed) so the project can be shared
> publicly. The method is exactly the one used on the real data.

## Method

### 1. Cleaning
Exact duplicates are dropped, non-positive amounts removed, and extreme
amount outliers (above the 99.9th percentile — data-entry errors) excluded.
Sample run: 2,391 raw rows → **2,360 valid transactions**, **650 customers**.

### 2. RFM scores
For each customer, at the snapshot date (day after the last transaction):

- **Recency** — days since the last purchase (lower is better)
- **Frequency** — number of distinct orders
- **Monetary** — total amount spent

Each dimension is scored **1–5 by quintiles**. Scores are computed on ranks
(`rank(method="first")`) so the many tied Frequency values cannot break the
quintile cuts.

### 3. Segments
Segments are assigned from the R and F scores with ordered rules:

| Segment | Rule (R, F scores) | Customers | Revenue share | Avg simplified CLV |
|---|---|---|---|---|
| Champions | R ≥ 4, F ≥ 4 | 178 (27.4%) | 53.9% | 1,818 |
| Loyal | R ≥ 3, F ≥ 3 | 166 (25.5%) | 27.1% | 981 |
| At Risk | R ≤ 2, F ≥ 2 | 140 (21.5%) | 11.1% | 476 |
| Lost | R = 1, F = 1 | 81 (12.5%) | 3.3% | 244 |
| Hibernating | R = 2, F = 1 | 39 (6.0%) | 1.4% | 216 |
| Promising | R = 3, F ≤ 2 | 25 (3.8%) | 1.5% | 353 |
| Potential Loyalists | R ≥ 4, F = 2–3 | 21 (3.2%) | 1.7% | 481 |

*(figures from the sample run — see `output/segment_summary.csv`, which also
contains one recommended marketing action per segment: reward the Champions,
win back the At Risk before they become Lost, onboard the new customers…)*

Key reading: **Champions + Loyal = 53% of customers but 81% of revenue**,
while the At Risk segment (21.5% of customers, historically good buyers gone
quiet) is the first reactivation target.

### 4. Simplified CLV
`CLV = average basket × orders per year × 2-year horizon`, computed per
customer and averaged per segment. It is deliberately simple — a ranking
tool for marketing priorities, not a predictive model.

### 5. Cohort analysis
Customers are grouped by **first-purchase month** (their cohort). The
retention matrix in `output/cohort_retention.csv` gives, for each cohort,
the share of customers still active N months after their first purchase
(cohort index 0 = 100% by construction). It shows whether recent cohorts
retain better or worse than older ones.

## Repository layout

```
segmentation-rfm-cohortes/
├── data/sample/transactions.csv   synthetic e-commerce transactions
├── src/
│   ├── generate_sample_data.py    regenerates the sample dataset (seeded)
│   └── rfm_cohort_analysis.py     clean → RFM + segments + CLV → cohorts
└── output/
    ├── rfm_segments.csv           one row per customer (scores + segment)
    ├── segment_summary.csv        segment KPIs + recommended action
    └── cohort_retention.csv       retention matrix in %
```

## How to run

```bash
pip install -r requirements.txt
python src/generate_sample_data.py   # optional: rebuild the sample data
python src/rfm_cohort_analysis.py    # writes all outputs to output/
```

## Using the outputs in Power BI

Import the three CSVs from `output/`: a segment overview page (customer and
revenue share by segment, average CLV), a customer table filtered by
segment, and the cohort matrix as a heatmap (cohort month × cohort index,
values = retention %). Slicing by segment makes the marketing actions page
actionable segment by segment.

## Tech stack

Python (pandas, NumPy) · SQL (data preparation upstream) · Power BI · Excel.
