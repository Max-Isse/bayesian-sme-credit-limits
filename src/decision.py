"""Stage 6: turn posterior estimates into a starting credit limit per segment.

The rule, per activated account over one year
---------------------------------------------
A business needs to put S pounds a month through its card. The card can
only carry up to its limit L, so the spend on the card is min(S, L).

    card spend   = E[min(S, L)]                        (monthly, S ~ lognormal)
    revenue      = card spend x (12 x interchange + net interest margin x share revolved)
    loss         = PD x LGD x exposure share at default x L

Raising L earns more only from businesses whose spend actually reaches it,
but expected loss grows with every pound of limit. So expected profit rises,
peaks and falls: there is a best limit.

Decision: choose L to maximise posterior expected profit, subject to

    P(loss > 50% of revenue) < 5%

where the probability is over the posterior draws of PD. Uncertainty in PD
therefore pulls limits down, and they rise as the posterior narrows.

Expected profit per approved account also includes activation and a fixed
cost:  P(activate) x (revenue - loss) - fixed cost.

All the economic numbers below are assumptions, chosen to be plausible for a
UK SME card, not market data. Change them in ASSUMPTIONS.

Run after sequential.py:  python src/decision.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm

from data_generation import TRUE_PARAMS
from plotting import COLORS, set_style

ROOT = Path(__file__).resolve().parents[1]
DRAWS_DIR = ROOT / "results" / "sequential"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = ROOT / "figures"

ASSUMPTIONS = {
    "interchange_rate": 0.015,  # 1.5% of card spend (commercial cards are not capped in the UK)
    "apr": 0.24,  # interest rate on revolving balances
    "funding_cost": 0.05,  # lender's cost of funding those balances
    "share_revolved": 0.30,  # average revolving balance as a share of monthly card spend
    "lgd": 0.85,  # loss given default: share of the exposure that is lost
    "exposure_share_at_default": 0.90,  # businesses draw up to ~90% of their limit before defaulting
    "fixed_cost": 50.0,  # yearly cost per approved account (onboarding, servicing), GBP
    "loss_to_revenue_threshold": 0.50,  # expected loss must stay below 50% of revenue...
    "max_breach_probability": 0.05,  # ...with at least 95% posterior probability
}
LIMIT_GRID = np.arange(250, 50_001, 250)  # candidate limits, GBP
MODELS = {"partial": "Hierarchical", "none": "No pooling"}
MODEL_COLORS = {"partial": COLORS["hierarchical"], "none": COLORS["no_pooling"], "oracle": COLORS["truth"]}
MONTHS = range(1, TRUE_PARAMS["n_cohorts"] + 1)
SMALL_SEGMENT = "Hospitality | Medium"
LARGE_SEGMENT = "Construction | Micro"


def revenue_per_pound_of_spend(a=ASSUMPTIONS):
    """Yearly revenue per pound of monthly card spend."""
    return 12 * a["interchange_rate"] + (a["apr"] - a["funding_cost"]) * a["share_revolved"]


def loss_per_pound_of_limit(pd_rate, a=ASSUMPTIONS):
    return pd_rate * a["lgd"] * a["exposure_share_at_default"]


def expected_card_spend(limit, mu, sigma):
    """E[min(S, L)] for S ~ LogNormal(mu, sigma), the standard closed form."""
    limit = np.asarray(limit, dtype=float)
    z = (np.log(limit) - mu) / sigma
    mean_s = np.exp(mu + sigma**2 / 2)
    return mean_s * norm.cdf(z - sigma) + limit * (1 - norm.cdf(z))


def spend_distribution(accounts):
    """Lognormal spend per segment, estimated from activated accounts.

    mu is each segment's own log median spend; sigma is pooled across the
    portfolio. Uncertainty in spend is ignored: this project is about the
    uncertainty in default and activation rates.
    """
    act = accounts[(accounts["activated_60d"] == 1) & (accounts["avg_monthly_spend"] > 0)].copy()
    act["log_spend"] = np.log(act["avg_monthly_spend"])
    mu = act.groupby("segment")["log_spend"].median()
    sigma = (act["log_spend"] - act["segment"].map(mu)).std()
    return mu, float(sigma)


def true_spend_distribution(segments):
    """The generator's true spend distribution (used only to score decisions)."""
    p = TRUE_PARAMS
    mu = pd.Series(
        [np.log(p["median_monthly_spend"][s] * p["industry_spend_multiplier"][i]) for i, s in
         zip(segments["industry"], segments["size_band"])],
        index=segments["segment"],
    )
    return mu, p["spend_log_sd"]


def evaluate_limits(pd_draws, pa_draws, mu, sigma, a=ASSUMPTIONS, grid=LIMIT_GRID):
    """Expected profit and breach probability for every candidate limit.

    pd_draws, pa_draws: posterior draws of one segment's default and
    activation rates (1-D arrays of the same length).
    """
    revenue = revenue_per_pound_of_spend(a) * expected_card_spend(grid, mu, sigma)  # (n_limits,)
    loss = loss_per_pound_of_limit(pd_draws[:, None], a) * grid[None, :]  # (n_draws, n_limits)
    profit = pa_draws[:, None] * (revenue[None, :] - loss) - a["fixed_cost"]
    breach = (loss > a["loss_to_revenue_threshold"] * revenue[None, :]).mean(axis=0)
    return profit.mean(axis=0), breach, revenue


def choose_limit(pd_draws, pa_draws, mu, sigma, a=ASSUMPTIONS, grid=LIMIT_GRID):
    """The profit-maximising limit among those that pass the risk rule.

    Returns limit 0 ("decline or refer") when no limit passes the rule or the
    best allowed limit loses money.
    """
    profit, breach, _ = evaluate_limits(pd_draws, pa_draws, mu, sigma, a, grid)
    allowed = breach < a["max_breach_probability"]
    unconstrained = grid[np.argmax(profit)]
    if not allowed.any():
        return {"limit": 0, "expected_profit": 0.0, "risk_rule_binds": True, "unconstrained_limit": unconstrained}
    i = np.flatnonzero(allowed)[np.argmax(profit[allowed])]
    if profit[i] <= 0:
        return {"limit": 0, "expected_profit": 0.0, "risk_rule_binds": False, "unconstrained_limit": unconstrained}
    return {
        "limit": int(grid[i]),
        "expected_profit": float(profit[i]),
        "risk_rule_binds": bool(grid[i] < unconstrained),
        "unconstrained_limit": int(unconstrained),
    }


def true_profit(limit, pd_true, pa_true, mu_true, sigma_true, a=ASSUMPTIONS):
    """What a limit actually earns per approved account under the true rates."""
    if limit == 0:
        return 0.0
    revenue = revenue_per_pound_of_spend(a) * expected_card_spend(limit, mu_true, sigma_true)
    return float(pa_true * (revenue - loss_per_pound_of_limit(pd_true, a) * limit) - a["fixed_cost"])


def load_draws(pooling, month):
    ds = xr.open_dataset(DRAWS_DIR / f"{pooling}_month_{month:02d}.nc")
    return ds["default_rate"].transpose("segment", "sample"), ds["activation_rate"].transpose("segment", "sample")


def limits_for_month(pooling, month, mu, sigma):
    pd_all, pa_all = load_draws(pooling, month)
    rows = []
    for seg in pd_all["segment"].values:
        pd_draws = pd_all.sel(segment=seg).values
        pa_draws = pa_all.sel(segment=seg).values
        choice = choose_limit(pd_draws, pa_draws, mu[seg], sigma)
        rows.append({
            "month": month, "model": pooling, "segment": seg,
            "pd_mean": pd_draws.mean(), "pd_q95": np.quantile(pd_draws, 0.95),
            "activation_mean": pa_draws.mean(), **choice,
        })
    return rows


def oracle_limits(truth, mu, sigma):
    """Limits chosen with the true rates known exactly (no uncertainty)."""
    rows = []
    for _, t in truth.iterrows():
        choice = choose_limit(np.array([t["true_default_rate"]]), np.array([t["true_activation_rate"]]),
                              mu[t["segment"]], sigma)
        rows.append({"model": "oracle", "segment": t["segment"], **choice})
    return pd.DataFrame(rows)


# Figures -------------------------------------------------------------------


def plot_rule_example(pooling, month, segment, mu, sigma, path):
    """One segment: expected profit and breach probability against the limit."""
    pd_all, pa_all = load_draws(pooling, month)
    pd_draws, pa_draws = pd_all.sel(segment=segment).values, pa_all.sel(segment=segment).values
    profit, breach, _ = evaluate_limits(pd_draws, pa_draws, mu[segment], sigma)
    choice = choose_limit(pd_draws, pa_draws, mu[segment], sigma)
    grid = LIMIT_GRID / 1000

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(grid, profit, color=COLORS["hierarchical"], lw=2)
    ax.axvline(choice["unconstrained_limit"] / 1000, color=COLORS["muted"], ls="--", lw=1,
               label=f"Highest expected profit: £{choice['unconstrained_limit']:,}")
    ax.axvline(choice["limit"] / 1000, color=COLORS["truth"], lw=1.5, label=f"Chosen limit: £{choice['limit']:,}")
    ax.set_xlabel("Credit limit (£ thousands)")
    ax.set_ylabel("Expected profit per approved account (£/year)")
    ax.set_title("Expected profit", loc="left")
    ax.legend(loc="lower left")
    ax = axes[1]
    ax.plot(grid, breach, color=COLORS["observed"], lw=2)
    ax.axhline(ASSUMPTIONS["max_breach_probability"], color=COLORS["muted"], ls="--", lw=1, label="5% ceiling")
    ax.axvline(choice["limit"] / 1000, color=COLORS["truth"], lw=1.5)
    ax.set_xlabel("Credit limit (£ thousands)")
    ax.set_ylabel("P(expected loss > 50% of revenue)")
    ax.set_title("Risk rule", loc="left")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    x_max = max(4 * choice["unconstrained_limit"], 4000)
    for a in axes:
        a.set_xlim(0, x_max / 1000)
    visible = profit[LIMIT_GRID <= x_max]
    pad = 0.1 * (visible.max() - visible.min())
    axes[0].set_ylim(visible.min() - pad, visible.max() + pad)
    fig.suptitle(f"How the limit is chosen: {segment}, {MODELS[pooling].lower()} model, month {month}",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_limits_by_segment(final, truth, path):
    order = truth.sort_values("n_accounts", ascending=False)
    y = {s: i for i, s in enumerate(order["segment"])}
    offsets = {"oracle": -0.25, "none": 0.0, "partial": 0.25}
    labels = {"oracle": "Oracle (true rates known)", **MODELS}
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for model in ["oracle", "none", "partial"]:
        d = final[final["model"] == model]
        ax.plot(d["limit"] / 1000, d["segment"].map(y) + offsets[model], "o" if model != "oracle" else "D",
                color=MODEL_COLORS[model], ms=7, label=labels[model])
    ax.set_yticks(range(len(order)), [f"{s}  (n={n})" for s, n in zip(order["segment"], order["n_accounts"])])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel("Suggested starting limit (£ thousands); 0 = decline or refer")
    ax.grid(axis="y", visible=False)
    for i in range(len(order)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f0efec", lw=0, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3)
    ax.set_title("Suggested starting limits at month 12 (smallest segments at the bottom)", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_limits_over_time(monthly, oracle, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    for ax, seg in zip(axes, [SMALL_SEGMENT, LARGE_SEGMENT]):
        for pooling in MODELS:
            d = monthly[(monthly["model"] == pooling) & (monthly["segment"] == seg)]
            ax.plot(d["month"], d["limit"] / 1000, "o-", color=MODEL_COLORS[pooling], lw=2, ms=5, label=MODELS[pooling])
        o = oracle.loc[oracle["segment"] == seg, "limit"].iloc[0]
        ax.axhline(o / 1000, color=COLORS["truth"], ls=":", lw=1.5, label="Oracle (true rates known)")
        ax.set_title(seg, loc="left")
        ax.set_xlabel("End of month since launch")
        ax.set_xticks(list(MONTHS))
        ax.set_ylabel("Suggested limit (£ thousands)")
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper left")
    fig.suptitle("How suggested limits rise as uncertainty shrinks", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def limit_vs_uncertainty(pooling, month, segment, mu, sigma, scales=np.linspace(0, 2, 21)):
    """What-if: keep a segment's posterior mean, scale its uncertainty.

    Draws are scaled around their mean on the log-odds scale, so a scale of
    0 means "rate known exactly" and 2 means "twice as uncertain as now".
    This separates the effect of uncertainty from the effect of the estimate
    itself moving, which the month-by-month results mix together.
    """
    pd_all, pa_all = load_draws(pooling, month)
    pd_draws, pa_draws = pd_all.sel(segment=segment).values, pa_all.sel(segment=segment).values
    logit = np.log(pd_draws / (1 - pd_draws))
    centre = logit.mean()
    rows = []
    for k in scales:
        scaled = 1 / (1 + np.exp(-(centre + k * (logit - centre))))
        width = np.quantile(scaled, 0.95) - np.quantile(scaled, 0.05)
        rows.append({"segment": segment, "scale": k, "interval_width": width,
                     "limit": choose_limit(scaled, pa_draws, mu[segment], sigma)["limit"]})
    return pd.DataFrame(rows)


def plot_limit_vs_uncertainty(whatif, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Colours here identify segments, so they avoid the model colours used elsewhere.
    for seg, color in [(SMALL_SEGMENT, "#6250d6"), (LARGE_SEGMENT, "#c98500")]:
        d = whatif[whatif["segment"] == seg]
        ax.plot(d["interval_width"] * 100, d["limit"] / 1000, "o-", color=color, lw=2, ms=4, label=seg)
        now = d[np.isclose(d["scale"], 1.0)].iloc[0]
        ax.plot(now["interval_width"] * 100, now["limit"] / 1000, "o", color=color, ms=11, mfc="none", mew=2)
    ax.plot([], [], "o", color=COLORS["muted"], ms=11, mfc="none", mew=2, label="Month-12 uncertainty")
    ax.set_xlabel("Width of 90% interval for the default rate (percentage points)")
    ax.set_ylabel("Suggested limit (£ thousands)")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set_title("What-if: same estimated default rate, more or less certainty", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    set_style()
    REPORTS_DIR.mkdir(exist_ok=True)
    accounts = pd.read_csv(ROOT / "data" / "accounts.csv")
    truth = pd.read_csv(ROOT / "data" / "true_segment_rates.csv")
    mu, sigma = spend_distribution(accounts)
    mu_true, sigma_true = true_spend_distribution(truth)

    monthly = pd.DataFrame([row for m in MONTHS for p in MODELS for row in limits_for_month(p, m, mu, sigma)])
    monthly.to_csv(REPORTS_DIR / "decision_limits_by_month.csv", index=False)
    oracle = oracle_limits(truth, mu, sigma)

    # Month-12 decisions, scored against the true economics.
    final = pd.concat([monthly[monthly["month"] == 12], oracle], ignore_index=True)
    t = truth.set_index("segment")
    final["true_profit_per_account"] = [
        true_profit(r["limit"], t.loc[r["segment"], "true_default_rate"], t.loc[r["segment"], "true_activation_rate"],
                    mu_true[r["segment"]], sigma_true)
        for _, r in final.iterrows()
    ]
    final["n_accounts"] = final["segment"].map(t["n_accounts"])
    final["true_profit_segment"] = final["true_profit_per_account"] * final["n_accounts"]
    final.to_csv(REPORTS_DIR / "decision_limits_month12.csv", index=False)

    plot_rule_example("partial", 12, LARGE_SEGMENT, mu, sigma, FIG_DIR / "decision_rule_example.png")
    plot_limits_by_segment(final, truth, FIG_DIR / "decision_limits_by_segment.png")
    plot_limits_over_time(monthly, oracle, FIG_DIR / "decision_limits_over_time.png")
    whatif = pd.concat([limit_vs_uncertainty("partial", 12, s, mu, sigma) for s in [SMALL_SEGMENT, LARGE_SEGMENT]])
    whatif.to_csv(REPORTS_DIR / "decision_limit_vs_uncertainty.csv", index=False)
    plot_limit_vs_uncertainty(whatif, FIG_DIR / "decision_limit_vs_uncertainty.png")

    pd.set_option("display.width", 200)
    wide = final.pivot(index="segment", columns="model", values="limit")
    wide = wide.join(final[final["model"] == "partial"].set_index("segment")[["n_accounts", "pd_mean", "pd_q95", "risk_rule_binds"]])
    nopool = final[final["model"] == "none"].set_index("segment")
    wide["pd_mean_none"], wide["pd_q95_none"] = nopool["pd_mean"], nopool["pd_q95"]
    wide = wide.join(t["true_default_rate"])
    print(f"Spend: pooled log-sd {sigma:.2f} (true {sigma_true})")
    print(wide.sort_values("n_accounts", ascending=False).round(3).to_string())
    print("\nTrue yearly profit of the month-12 limits, whole portfolio (£):")
    print(final.groupby("model")["true_profit_segment"].sum().round(0).to_string())
    print("\nWhat-if (hierarchical, month 12): limit as uncertainty is scaled:")
    print(whatif.pivot(index="scale", columns="segment", values=["interval_width", "limit"]).iloc[::4].round(3).to_string())
    print("\nLimits by month (£):")
    print(monthly[monthly["segment"].isin([SMALL_SEGMENT, LARGE_SEGMENT])]
          .pivot_table(index=["segment", "model"], columns="month", values="limit").to_string())


if __name__ == "__main__":
    main()
