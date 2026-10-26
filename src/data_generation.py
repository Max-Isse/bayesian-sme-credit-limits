"""Simulate a newly launched SME business credit card portfolio.

Everything here is simulated. The true parameters are fixed in TRUE_PARAMS and
saved to disk, so later stages can check whether the models recover them.

Generative story for one account
--------------------------------
1. The business joins in a monthly cohort (month 1..12) and belongs to one of
   12 segments (4 industries x 3 size bands).
2. It activates the card within 60 days with its segment's activation rate.
3. If (and only if) it activated, it defaults within its first 12 months with
   its segment's default rate. A card that is never used carries no balance,
   so it cannot default. The default rate is therefore P(default | activated).
4. If it defaults, the month of default is uniform on 1..12.
5. We take a snapshot at the end of `snapshot_month`. Later cohorts have had
   less time on book, so some of their future defaults are not yet visible
   (right-censoring). The observed data only show what was visible at the
   snapshot; the full truth is written to a separate file.

Run:  python src/data_generation.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

INDUSTRIES = ["Construction", "Retail", "Professional Services", "Hospitality"]
SIZE_BANDS = ["Micro", "Small", "Medium"]

TRUE_PARAMS = {
    "seed_segment_effects": 2024,
    "seed_accounts": 7,
    "n_cohorts": 12,
    # Total accounts per segment over the 12 months. Deliberately unbalanced:
    # two segments have fewer than 30 accounts, several have hundreds.
    "segment_sizes": {
        "Construction": {"Micro": 600, "Small": 260, "Medium": 70},
        "Retail": {"Micro": 520, "Small": 210, "Medium": 28},
        "Professional Services": {"Micro": 450, "Small": 300, "Medium": 90},
        "Hospitality": {"Micro": 240, "Small": 80, "Medium": 22},
    },
    # Applications grow as the product ramps up after launch: cohort k gets a
    # share proportional to (1 + growth * (k - 1)).
    "cohort_growth": 0.15,
    # Segment rates are built on the log-odds (logit) scale:
    #   logit(rate) = intercept + industry effect + size effect + segment noise
    # The noise makes each segment slightly different from what industry and
    # size alone would predict, as real segments are.
    "default": {
        "intercept": -2.944,  # logit(0.05): a 5% baseline 12-month default rate
        "industry_effect": {
            "Construction": 0.35,
            "Retail": 0.10,
            "Professional Services": -0.45,
            "Hospitality": 0.60,
        },
        "size_effect": {"Micro": 0.30, "Small": 0.0, "Medium": -0.35},
        "segment_noise_sd": 0.15,
    },
    "activation": {
        "intercept": 0.847,  # logit(0.70): a 70% baseline activation rate
        "industry_effect": {
            "Construction": -0.10,
            "Retail": 0.0,
            "Professional Services": 0.30,
            "Hospitality": -0.30,
        },
        "size_effect": {"Micro": -0.30, "Small": 0.0, "Medium": 0.35},
        "segment_noise_sd": 0.15,
    },
    # Spend and limits (GBP). Spend is lognormal around a median set by size
    # band and scaled by industry.
    "median_monthly_spend": {"Micro": 1200, "Small": 3500, "Medium": 9000},
    "industry_spend_multiplier": {
        "Construction": 1.2,
        "Retail": 1.0,
        "Professional Services": 0.8,
        "Hospitality": 0.9,
    },
    "spend_log_sd": 0.6,
    # The simple launch policy: one starting limit per size band.
    "launch_credit_limit": {"Micro": 2500, "Small": 6000, "Medium": 15000},
    # Utilisation (month-end balance / limit) ~ Beta. Businesses heading for
    # default tend to run their balances up, so they get a higher mean.
    "utilisation_beta_performing": [3.0, 7.0],  # mean 0.30
    "utilisation_beta_defaulter": [6.0, 2.0],  # mean 0.75
}


def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def segment_table(params=TRUE_PARAMS):
    """Return one row per segment with its true default and activation rates."""
    rng = np.random.default_rng(params["seed_segment_effects"])
    rows = []
    for industry in INDUSTRIES:
        for size in SIZE_BANDS:
            rows.append({"industry": industry, "size_band": size})
    seg = pd.DataFrame(rows)
    seg["segment"] = seg["industry"] + " | " + seg["size_band"]
    seg["n_accounts"] = [
        params["segment_sizes"][i][s] for i, s in zip(seg["industry"], seg["size_band"])
    ]

    for outcome in ["default", "activation"]:
        p = params[outcome]
        noise = rng.normal(0.0, p["segment_noise_sd"], size=len(seg))
        logit = (
            p["intercept"]
            + seg["industry"].map(p["industry_effect"])
            + seg["size_band"].map(p["size_effect"])
            + noise
        )
        seg[f"{outcome}_noise"] = noise
        seg[f"true_{outcome}_logit"] = logit
        seg[f"true_{outcome}_rate"] = inv_logit(logit)
    return seg


def simulate_portfolio(params=TRUE_PARAMS, seed=None, snapshot_month=12, size_multiplier=1.0):
    """Simulate every account and what is visible at the end of `snapshot_month`.

    Returns (observed, truth, segments):
      observed: what the lender can see at the snapshot (use this for modelling)
      truth:    the full 12-month outcomes, including defaults not yet visible
      segments: the true segment-level rates

    `seed` defaults to params["seed_accounts"]. A different seed (and a larger
    `snapshot_month`) gives an independent hold-out portfolio with the same
    true rates, which Stage 4 uses for calibration.
    """
    seed = params["seed_accounts"] if seed is None else seed
    rng = np.random.default_rng(seed)
    seg = segment_table(params)
    n_cohorts = params["n_cohorts"]

    cohort_weights = 1 + params["cohort_growth"] * np.arange(n_cohorts)
    cohort_weights = cohort_weights / cohort_weights.sum()

    frames = []
    for _, s in seg.iterrows():
        n = int(round(s["n_accounts"] * size_multiplier))
        cohort = rng.choice(np.arange(1, n_cohorts + 1), size=n, p=cohort_weights)
        frames.append(
            pd.DataFrame(
                {
                    "industry": s["industry"],
                    "size_band": s["size_band"],
                    "segment": s["segment"],
                    "cohort_month": cohort,
                    "true_p_activation": s["true_activation_rate"],
                    "true_p_default": s["true_default_rate"],
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["cohort_month", "segment"], kind="stable").reset_index(drop=True)
    df.insert(0, "account_id", [f"A{i:05d}" for i in range(1, len(df) + 1)])
    n = len(df)

    # Months on book at the snapshot: cohort 1 joined in month 1, so at the end
    # of month 12 it has 12 months; cohort 12 has 1 month.
    df["months_on_book"] = snapshot_month - df["cohort_month"] + 1

    # True outcomes (what eventually happens).
    activated = rng.random(n) < df["true_p_activation"].to_numpy()
    defaults = activated & (rng.random(n) < df["true_p_default"].to_numpy())
    default_month = np.where(defaults, rng.integers(1, 13, size=n), 0)

    # Limits and spend.
    df["credit_limit"] = df["size_band"].map(params["launch_credit_limit"]).astype(float)
    median_spend = df["size_band"].map(params["median_monthly_spend"]) * df["industry"].map(
        params["industry_spend_multiplier"]
    )
    spend = median_spend.to_numpy() * np.exp(rng.normal(0, params["spend_log_sd"], size=n))
    a_perf, b_perf = params["utilisation_beta_performing"]
    a_def, b_def = params["utilisation_beta_defaulter"]
    utilisation = np.where(
        defaults, rng.beta(a_def, b_def, size=n), rng.beta(a_perf, b_perf, size=n)
    )
    # A card that was never activated has no spend and no balance.
    spend = np.where(activated, spend, 0.0)
    utilisation = np.where(activated, utilisation, 0.0)

    truth = pd.DataFrame(
        {
            "account_id": df["account_id"],
            "activated_60d_true": activated.astype(int),
            "default_12m_true": defaults.astype(int),
            "default_month_true": np.where(defaults, default_month, np.nan),
        }
    )

    # What is visible at the snapshot.
    mob = df["months_on_book"].to_numpy()
    activation_known = mob >= 2  # the 60-day window has closed
    default_seen = defaults & (default_month <= mob)

    observed = df.drop(columns=["true_p_activation", "true_p_default"]).copy()
    # Unknown (<NA>) for accounts whose 60-day window is still open.
    observed["activated_60d"] = pd.Series(np.where(activation_known, activated, np.nan)).astype(
        "Int64"
    )
    # Too early to report spend and utilisation for accounts under 2 months old.
    observed["avg_monthly_spend"] = np.where(activation_known, spend.round(2), np.nan)
    observed["utilisation"] = np.where(activation_known, utilisation.round(4), np.nan)
    observed["defaulted"] = default_seen.astype(int)
    observed["default_month"] = np.where(default_seen, default_month, np.nan)
    # Fraction of the 12-month default window observed so far.
    observed["exposure_fraction"] = np.minimum(mob, 12) / 12

    return observed, truth, seg


def save_outputs(out_dir="data", params=TRUE_PARAMS):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    observed, truth, seg = simulate_portfolio(params)
    observed.to_csv(out / "accounts.csv", index=False)
    truth.to_csv(out / "accounts_truth.csv", index=False)
    seg.to_csv(out / "true_segment_rates.csv", index=False)
    with open(out / "true_parameters.json", "w") as f:
        json.dump(
            {
                "generator_settings": params,
                "true_segment_rates": seg[
                    ["segment", "n_accounts", "true_default_rate", "true_activation_rate"]
                ].to_dict(orient="records"),
            },
            f,
            indent=2,
        )
    return observed, truth, seg


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    observed, truth, seg = save_outputs(project_root / "data")
    print(f"Accounts: {len(observed):,}")
    print(seg[["segment", "n_accounts", "true_default_rate", "true_activation_rate"]].round(3))
