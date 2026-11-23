"""
Generate the synthetic e-commerce transactions dataset used by this project.

100% fictional data: ~650 customers over 18 months (Jan 2024 - Jun 2025).
Each customer gets an individual purchase rhythm (Poisson rate drawn from a
gamma distribution), which produces a realistic mix of one-time buyers,
regular customers and heavy buyers for the RFM segmentation. A few dirty
rows (duplicates, a negative amount, an extreme outlier) are injected on
purpose so the cleaning step has something real to do.

Usage:
    python src/generate_sample_data.py
Output:
    data/sample/transactions.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 650
START = np.datetime64("2024-01-01")
END = np.datetime64("2025-06-30")


def main() -> None:
    rng = np.random.default_rng(SEED)
    n_days = int((END - START).astype(int)) + 1

    rows = []
    order_seq = 0
    for customer in range(1, N_CUSTOMERS + 1):
        customer_id = f"CUST-{customer:04d}"
        # Individual purchase rhythm: expected orders over the whole period.
        lam = rng.gamma(shape=1.6, scale=2.2)
        n_orders = max(1, rng.poisson(lam))
        # First purchase spread over the first 15 months (cohorts).
        first_day = int(rng.integers(0, int(n_days * 0.85)))
        days = np.sort(rng.integers(first_day, n_days, size=n_orders))
        days[0] = first_day
        for day in days:
            order_seq += 1
            # Basket value: log-normal, slightly higher for frequent buyers.
            amount = float(rng.lognormal(mean=5.1, sigma=0.55))
            rows.append(
                {
                    "order_id": f"ORD-{order_seq:06d}",
                    "customer_id": customer_id,
                    "order_date": str(START + int(day)),
                    "amount": round(amount, 2),
                }
            )

    df = pd.DataFrame(rows)

    # --- Inject a few dirty rows on purpose (cleaning demo) -----------------
    duplicates = df.sample(25, random_state=SEED)
    bad_amount = df.sample(5, random_state=SEED + 1).copy()
    bad_amount["amount"] = [-120.0, 0.0, 999_999.0, -1.0, 888_888.0]
    bad_amount["order_id"] = [f"ORD-BAD-{i:02d}" for i in range(5)]
    df = pd.concat([df, duplicates, bad_amount], ignore_index=True)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    out = Path(__file__).resolve().parents[1] / "data" / "sample" / "transactions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows -> {out}")


if __name__ == "__main__":
    main()
