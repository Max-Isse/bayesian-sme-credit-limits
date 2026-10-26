"""Stage 2: prior predictive checks.

A prior predictive check asks: before seeing any outcomes, what does the
model think the world looks like? We draw parameters from the priors, simulate
data from them, and check the results are plausible for an SME card.

The checks run on the same hierarchical model that is fitted in Stage 3
(model.build_model), so what is checked here is what gets used.

Run:  python src/prior_checks.py
Writes figures/prior_*.png and prints summary numbers.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from model import build_model, prepare_data, segment_frame
from plotting import COLORS, PRIOR_COLORS, set_style
from priors import PRIORS, sample_flat_rates_numpy, sample_segment_rates_numpy

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
N_DRAWS = 4000
SEED = 11


def compare_candidate_priors(rng, n=100_000):
    """Implied segment default rates under three candidate priors."""
    seg = segment_frame()
    chosen = sample_segment_rates_numpy(
        PRIORS["default"], n // len(seg), rng, seg["industry_idx"].to_numpy(), seg["size_idx"].to_numpy()
    )
    return {
        "Vague: Normal(0, 10) on log-odds": sample_flat_rates_numpy(0.0, 10.0, n, rng),
        "Generic: Normal(0, 1.5) on log-odds": sample_flat_rates_numpy(0.0, 1.5, n, rng),
        "Chosen: hierarchical, centred on 5%": chosen.ravel(),
    }


def plot_candidate_priors(candidates, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
    bins = np.linspace(0, 1, 51)
    for ax, (label, draws), color in zip(axes, candidates.items(), PRIOR_COLORS):
        ax.axvspan(0, 0.20, color=COLORS["band"], alpha=0.6, lw=0, label="Plausible: 0-20%")
        ax.hist(draws, bins=bins, density=True, color=color, edgecolor="#fcfcfb", linewidth=0.8)
        share = np.mean(draws <= 0.20)
        ax.set_title(f"{label}\n{share:.0%} of draws in 0-20%", fontsize=10)
        ax.set_xlabel("Implied 12-month default rate")
        ax.set_yticks([])
    axes[0].legend(loc="upper center")
    fig.suptitle(
        "What each candidate prior believes about a segment's default rate, before seeing data",
        x=0.01, ha="left", fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_chosen_priors(prior, data, path):
    seg_rates = prior.prior["default_rate"].values.reshape(-1, len(data["segments"]))
    def_rates = seg_rates.ravel()
    act_rates = prior.prior["activation_rate"].values.ravel()
    # Riskiest vs safest segment within each draw: a check on the spread priors.
    spread_ratio = seg_rates.max(axis=1) / seg_rates.min(axis=1)
    pp_defaults = prior.prior_predictive["defaulted"].values.reshape(N_DRAWS, -1).sum(axis=1)
    pp_activated = prior.prior_predictive["activated"].values.reshape(N_DRAWS, -1).sum(axis=1)
    pp_activation_share = pp_activated / data["act_n"].sum()
    observed_defaults = int(data["def_k"].sum())

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    color = COLORS["hierarchical"]

    ax = axes[0, 0]
    ax.axvspan(0, 0.20, color=COLORS["band"], alpha=0.6, lw=0, label="Plausible: 0-20%")
    ax.hist(def_rates, bins=np.linspace(0, 0.5, 51), color=color, edgecolor="#fcfcfb", linewidth=0.8)
    ax.set_title("Segment default rates (all 12 segments, all draws)")
    ax.set_xlabel("12-month default rate given activation")
    ax.set_yticks([])
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    ax.hist(act_rates, bins=np.linspace(0, 1, 51), color=color, edgecolor="#fcfcfb", linewidth=0.8)
    ax.set_title("Segment activation rates")
    ax.set_xlabel("60-day activation rate")
    ax.set_yticks([])

    ax = axes[1, 0]
    ax.hist(np.clip(spread_ratio, 1, 20), bins=np.linspace(1, 20, 39), color=color,
            edgecolor="#fcfcfb", linewidth=0.8)
    ax.set_title("Riskiest / safest segment default rate (per draw)")
    ax.set_xlabel("Ratio (values above 20 shown at 20)")
    ax.set_yticks([])

    ax = axes[1, 1]
    ax.hist(pp_defaults, bins=np.arange(0, np.percentile(pp_defaults, 99.5) + 10, 5), color=color,
            edgecolor="#fcfcfb", linewidth=0.8)
    ax.axvline(observed_defaults, color=COLORS["observed"], lw=2)
    ax.text(observed_defaults, ax.get_ylim()[1] * 0.95,
            f"  Observed in simulated data: {observed_defaults}",
            color="#0b0b0b", fontsize=9, va="top")
    ax.set_title("Visible defaults this portfolio would show by month 12")
    ax.set_xlabel("Number of defaults (all segments)")
    ax.set_yticks([])

    fig.suptitle("Prior predictive checks for the chosen priors", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {
        "default_rate_quantiles": np.quantile(def_rates, [0.025, 0.25, 0.5, 0.75, 0.975]),
        "share_default_rate_le_20pct": np.mean(def_rates <= 0.20),
        "share_default_rate_gt_50pct": np.mean(def_rates > 0.50),
        "activation_rate_quantiles": np.quantile(act_rates, [0.025, 0.5, 0.975]),
        "spread_ratio_quantiles": np.quantile(spread_ratio, [0.05, 0.5, 0.95]),
        "pp_defaults_quantiles": np.quantile(pp_defaults, [0.025, 0.5, 0.975]),
        "pp_activation_share_quantiles": np.quantile(pp_activation_share, [0.025, 0.5, 0.975]),
        "observed_defaults": observed_defaults,
        "observed_activation_share": data["act_k"].sum() / data["act_n"].sum(),
    }


def main():
    set_style()
    FIG_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    accounts = pd.read_csv(ROOT / "data" / "accounts.csv")

    candidates = compare_candidate_priors(rng)
    plot_candidate_priors(candidates, FIG_DIR / "prior_candidates.png")
    print("Share of implied segment default rates within 0-20%:")
    for label, draws in candidates.items():
        print(f"  {label:40s} {np.mean(draws <= 0.20):.1%}")

    data = prepare_data(accounts)
    with build_model(data, pooling="partial"):
        prior = pm.sample_prior_predictive(draws=N_DRAWS, random_seed=SEED)

    summary = plot_chosen_priors(prior, data, FIG_DIR / "prior_predictive_checks.png")
    print("\nChosen priors:")
    for k, v in summary.items():
        print(f"  {k:32s} {np.round(v, 3)}")


if __name__ == "__main__":
    main()
