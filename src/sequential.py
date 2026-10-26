"""Stage 5: update the estimates month by month as new data arrives.

At the end of each month t we only see what was visible then: the cohorts
that have joined so far, each with t - cohort + 1 months on book. We refit
the model on everything visible at that date.

Why refit rather than "use last month's posterior as this month's prior"?
The same account keeps supplying new information as it ages: an account seen
for 5 months in May is seen for 6 months in June. Re-using May's posterior as
June's prior would count its first 5 months twice. Refitting on all data to
date is exact Bayesian updating with no double counting.

At the end of month 1 no account has passed its 60-day activation window,
so there is nothing to learn from yet: the month-1 estimate is the prior.

Run:  python src/sequential.py
Writes results/sequential/*.nc (posterior draws per month, used in Stage 6),
reports/sequential_summary.csv and figures/sequential_*.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from data_generation import TRUE_PARAMS, simulate_portfolio
from model import build_model, convergence_table, fit, n_divergences, prepare_data
from plotting import COLORS, set_style

ROOT = Path(__file__).resolve().parents[1]
DRAWS_DIR = ROOT / "results" / "sequential"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = ROOT / "figures"

MONTHS = range(1, TRUE_PARAMS["n_cohorts"] + 1)
MODELS = ["partial", "none"]
OUTCOMES = ["default", "activation"]
# 24 refits with PyMC's default sampler take ~7 minutes, mostly per-step
# overhead. The nutpie sampler (same NUTS algorithm, written in Rust) is
# several times faster, so we can afford more draws. Every fit's R-hat, ESS
# and divergences are recorded to check the result.
DRAWS, TUNE, SAMPLER = 2000, 1500, "nutpie"
SMALL_SEGMENT = "Hospitality | Medium"  # fewest accounts (22 by month 12)
LARGE_SEGMENT = "Construction | Micro"  # most accounts (600 by month 12)


def snapshot(month):
    """The portfolio as it looked at the end of `month`.

    The simulator's random draws do not depend on the snapshot date, so every
    snapshot shows the same accounts and outcomes, just less of them.
    """
    observed, _, _ = simulate_portfolio(snapshot_month=month)
    return observed[observed["cohort_month"] <= month].reset_index(drop=True)


def rate_draws(month, pooling):
    """Posterior (or, at month 1, prior) draws of every segment's rates.

    Returns (draws, diagnostics). Diagnostics are None for the prior.
    """
    data = prepare_data(snapshot(max(month, 2)))  # month 1: structure only, no outcomes used
    model = build_model(data, pooling)
    if month == 1:
        with model:
            idata = pm.sample_prior_predictive(draws=4000, random_seed=month)
        group, diagnostics = "prior", None
    else:
        idata = fit(model, draws=DRAWS, tune=TUNE, random_seed=month, nuts_sampler=SAMPLER)
        group = "posterior"
        table = convergence_table(idata)
        diagnostics = {
            "month": month, "model": pooling,
            "max_r_hat": table["r_hat"].max(),
            "min_ess_bulk": table["ess_bulk"].min(),
            "min_ess_tail": table["ess_tail"].min(),
            "divergences": n_divergences(idata),
        }
    draws = idata[group].to_dataset()[["default_rate", "activation_rate"]]
    return draws.stack(sample=("chain", "draw")).reset_index("sample", drop=True), diagnostics


def summarise(draws, month, pooling):
    rows = []
    for outcome in OUTCOMES:
        x = draws[f"{outcome}_rate"]
        q = x.quantile([0.05, 0.25, 0.5, 0.75, 0.95], dim="sample")
        for i, seg in enumerate(x["segment"].values):
            rows.append({
                "month": month, "model": pooling, "outcome": outcome, "segment": seg,
                "mean": float(x.isel(segment=i).mean()),
                **{f"q{int(p * 100):02d}": float(q.isel(segment=i).sel(quantile=p)) for p in q["quantile"].values},
            })
    return rows


def counts_by_month():
    """Accounts and visible defaults per segment at each month end."""
    rows = []
    for month in MONTHS:
        s = snapshot(month)
        g = s.groupby("segment").agg(accounts=("account_id", "size"), visible_defaults=("defaulted", "sum"))
        g["month"] = month
        rows.append(g.reset_index())
    return pd.concat(rows, ignore_index=True)


def plot_two_segments(summary, counts, truth, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    colors = {"partial": COLORS["hierarchical"], "none": COLORS["no_pooling"]}
    labels = {"partial": "Hierarchical", "none": "No pooling"}
    for col, seg in enumerate([SMALL_SEGMENT, LARGE_SEGMENT]):
        c = counts[counts["segment"] == seg].set_index("month")
        for row, outcome in enumerate(OUTCOMES):
            ax = axes[row, col]
            for pooling in ["none", "partial"]:
                d = summary.query("model == @pooling and outcome == @outcome and segment == @seg")
                ax.fill_between(d["month"], d["q05"], d["q95"], color=colors[pooling],
                                alpha=0.18 if pooling == "partial" else 0.0, lw=0)
                if pooling == "none":
                    ax.plot(d["month"], d["q05"], color=colors[pooling], lw=1.2, ls="--")
                    ax.plot(d["month"], d["q95"], color=colors[pooling], lw=1.2, ls="--",
                            label="No pooling: 90% interval")
                else:
                    ax.plot(d["month"], d["mean"], color=colors[pooling], lw=2, label="Hierarchical: mean")
                    ax.plot([], [], color=colors[pooling], alpha=0.3, lw=8, label="Hierarchical: 90% interval")
            true_rate = truth.loc[truth["segment"] == seg, f"true_{outcome}_rate"].iloc[0]
            ax.axhline(true_rate, color=COLORS["truth"], lw=1.2, ls=":", label="True rate")
            ax.set_ylim(0, 0.35 if outcome == "default" else 1.0)
            ax.set_ylabel("12-month default rate (axis cut at 35%)" if outcome == "default" else "60-day activation rate")
            if row == 0:
                ax.set_title(f"{seg}: {c.loc[12, 'accounts']} accounts and "
                             f"{c.loc[12, 'visible_defaults']} visible defaults by month 12", loc="left")
            if row == 1:
                ax.set_xlabel("End of month since launch")
                ax.set_xticks(list(MONTHS))
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("How the estimates narrow as monthly data arrives (month 1 = prior only)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def plot_width_all_segments(summary, truth, path):
    """90% interval width over time for every segment, hierarchical model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    n = truth.set_index("segment")["n_accounts"]
    for ax, outcome in zip(axes, OUTCOMES):
        d = summary.query("model == 'partial' and outcome == @outcome").copy()
        d["width"] = (d["q95"] - d["q05"]) * 100
        for seg, g in d.groupby("segment"):
            small = n[seg] < 100
            ax.plot(g["month"], g["width"], color=COLORS["no_pooling"] if small else COLORS["hierarchical"],
                    lw=1.5, alpha=0.8)
        ax.plot([], [], color=COLORS["no_pooling"], label="Small segment (< 100 accounts)")
        ax.plot([], [], color=COLORS["hierarchical"], label="Large segment")
        ax.set_title(f"{outcome.capitalize()}: width of 90% interval (percentage points)", loc="left")
        ax.set_xlabel("End of month since launch")
        ax.set_xticks(list(MONTHS))
        ax.set_ylim(bottom=0)
    axes[0].legend()
    fig.suptitle("Hierarchical model: uncertainty in every segment, month by month", x=0.01,
                 ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    set_style()
    DRAWS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    truth = pd.read_csv(ROOT / "data" / "true_segment_rates.csv")

    rows, diag_rows = [], []
    for month in MONTHS:
        for pooling in MODELS:
            draws, diagnostics = rate_draws(month, pooling)
            draws.to_netcdf(DRAWS_DIR / f"{pooling}_month_{month:02d}.nc")
            rows += summarise(draws, month, pooling)
            if diagnostics:
                diag_rows.append(diagnostics)
        print(f"month {month} done", flush=True)

    diag = pd.DataFrame(diag_rows)
    diag.to_csv(REPORTS_DIR / "sequential_diagnostics.csv", index=False)
    print(f"Monthly fits: max R-hat {diag['max_r_hat'].max():.3f}, min bulk ESS {diag['min_ess_bulk'].min():.0f}, "
          f"min tail ESS {diag['min_ess_tail'].min():.0f}, total divergences {diag['divergences'].sum()}")

    summary = pd.DataFrame(rows)
    summary.to_csv(REPORTS_DIR / "sequential_summary.csv", index=False)
    counts = counts_by_month()
    counts.to_csv(REPORTS_DIR / "sequential_counts.csv", index=False)
    plot_two_segments(summary, counts, truth, FIG_DIR / "sequential_two_segments.png")
    plot_width_all_segments(summary, truth, FIG_DIR / "sequential_interval_width.png")

    w = summary.assign(width=summary["q95"] - summary["q05"])
    for seg in [SMALL_SEGMENT, LARGE_SEGMENT]:
        t = w[(w["segment"] == seg) & (w["outcome"] == "default")].pivot(index="month", columns="model", values="width")
        print(f"\n{seg}: width of 90% default interval (pp)")
        print((t * 100).round(1).T.to_string())


if __name__ == "__main__":
    main()
