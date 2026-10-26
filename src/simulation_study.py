"""Optional: repeat the Stage 4 validation over many simulated datasets.

Stage 4 used one simulated portfolio (seed 7). Any single dataset can be
lucky or unlucky, so here we repeat generate -> fit -> score for many seeds
and look at the average behaviour of each model.

By default the true segment rates stay fixed (the ones in TRUE_PARAMS) and
only the accounts are re-drawn. That asks: "for this portfolio, how do the
models behave across the datasets we might have seen?" With
--vary-truth, each seed also draws new segment effects, which asks the more
general "does the method work for portfolios like this one?"

This takes roughly 1-1.5 minutes per seed, so it is not part of the main
pipeline. Results are appended seed by seed, so an interrupted run resumes.

Run:  python src/simulation_study.py --seeds 20
      python src/simulation_study.py --summarise-only
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_generation import TRUE_PARAMS, simulate_portfolio
from model import POOLING_TYPES, build_model, fit, prepare_data
from plotting import COLORS, set_style
from validate import (
    HOLDOUT,
    INTERVAL,
    MODEL_COLORS,
    MODEL_LABELS,
    SMALL_SEGMENT,
    calibration,
    recovery_table,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
FIG_DIR = ROOT / "figures"
FIRST_SEED = 1000  # well away from the seeds used elsewhere (7, 101, 2025)


def run_one_seed(seed, vary_truth):
    params = {**TRUE_PARAMS, "seed_segment_effects": seed} if vary_truth else TRUE_PARAMS
    observed, _, truth = simulate_portfolio(params, seed=seed)
    data = prepare_data(observed)
    fits = {p: fit(build_model(data, p), random_seed=seed) for p in POOLING_TYPES}

    rec = recovery_table(fits, truth)
    rec["seed"] = seed

    holdout, _, _ = simulate_portfolio(params, seed=seed + 50_000, **{k: v for k, v in HOLDOUT.items() if k != "seed"})
    holdout["activated_60d"] = holdout["activated_60d"].astype(int)
    small = truth.loc[truth["n_accounts"] < SMALL_SEGMENT, "segment"]
    _, metrics = calibration(fits, holdout, small)
    metrics["seed"] = seed
    return rec, metrics


def run(n_seeds, vary_truth, tag):
    REPORTS_DIR.mkdir(exist_ok=True)
    rec_path = REPORTS_DIR / f"simulation_study{tag}_recovery.csv"
    met_path = REPORTS_DIR / f"simulation_study{tag}_holdout.csv"
    done = set(pd.read_csv(rec_path)["seed"]) if rec_path.exists() else set()

    for seed in range(FIRST_SEED, FIRST_SEED + n_seeds):
        if seed in done:
            continue
        rec, metrics = run_one_seed(seed, vary_truth)
        rec.to_csv(rec_path, mode="a", header=not rec_path.exists(), index=False)
        metrics.to_csv(met_path, mode="a", header=not met_path.exists(), index=False)
        print(f"seed {seed} done", flush=True)


def summarise(tag):
    rec = pd.read_csv(REPORTS_DIR / f"simulation_study{tag}_recovery.csv")
    met = pd.read_csv(REPORTS_DIR / f"simulation_study{tag}_holdout.csv")
    rec["size_group"] = np.where(rec["n_accounts"] < SMALL_SEGMENT, "small", "large")
    rec["error"] = rec["mean"] - rec["true"]

    # Per seed first, then averaged over seeds, so every dataset counts equally.
    per_seed = (
        rec.groupby(["seed", "outcome", "model", "size_group"])
        .agg(mae=("abs_error", "mean"), coverage=("covered", "mean"), width=("interval_width", "mean"))
        .reset_index()
    )
    summary = (
        per_seed.groupby(["outcome", "model", "size_group"])
        .agg(seeds=("seed", "nunique"), mean_abs_error=("mae", "mean"),
             coverage=("coverage", "mean"), mean_interval_width=("width", "mean"))
        .reset_index()
    )

    # Head to head: in how many datasets does the hierarchical model beat no pooling?
    wide = per_seed.pivot_table(index=["seed", "outcome", "size_group"], columns="model", values="mae")
    h2h = (
        (wide["partial"] < wide["none"]).groupby(["outcome", "size_group"]).mean()
        .rename("share_of_seeds_hierarchical_more_accurate").reset_index()
    )
    ll = met.pivot_table(index=["seed", "outcome", "segments"], columns="model", values="log_loss")
    ll_h2h = (
        (ll["partial"] < ll["none"]).groupby(["outcome", "segments"]).mean()
        .rename("share_of_seeds_hierarchical_lower_log_loss").reset_index()
    )
    ll_mean = met.groupby(["outcome", "segments", "model"])["log_loss"].mean().unstack("model").reset_index()

    # Systematic bias per segment: average (estimate - truth) across seeds.
    bias = rec.groupby(["outcome", "model", "segment", "n_accounts"])["error"].mean().reset_index()

    summary.to_csv(REPORTS_DIR / f"simulation_study{tag}_summary.csv", index=False)
    bias.to_csv(REPORTS_DIR / f"simulation_study{tag}_bias_by_segment.csv", index=False)
    return per_seed, summary, h2h, ll_h2h, ll_mean, bias


def plot_study(per_seed, bias, n_seeds, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    groups = [("default", "small"), ("default", "large"), ("activation", "small"), ("activation", "large")]
    offsets = {"complete": -0.25, "none": 0.0, "partial": 0.25}
    rng = np.random.default_rng(0)

    for col, (metric, label, ref) in enumerate([
        ("mae", "Mean absolute error (percentage points)", None),
        ("coverage", f"Share of true rates inside the {INTERVAL:.0%} interval (dashed: target)", INTERVAL),
    ]):
        ax = axes[0, col]
        for pooling in POOLING_TYPES:
            for i, (outcome, size) in enumerate(groups):
                v = per_seed.query("model == @pooling and outcome == @outcome and size_group == @size")[metric]
                v = v * 100 if metric == "mae" else v
                x = i + offsets[pooling]
                ax.plot(x + rng.uniform(-0.06, 0.06, len(v)), v, "o", color=MODEL_COLORS[pooling],
                        alpha=0.35, ms=4)
                ax.plot([x - 0.1, x + 0.1], [v.mean()] * 2, color=MODEL_COLORS[pooling], lw=3,
                        label=MODEL_LABELS[pooling] if i == 0 else None)
        if ref is not None:
            ax.axhline(ref, color=COLORS["muted"], ls="--", lw=1)
        ax.set_xticks(range(len(groups)), [f"{o.capitalize()}\n({s})" for o, s in groups])
        ax.set_title(label, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)

    for col, outcome in enumerate(["default", "activation"]):
        ax = axes[1, col]
        b = bias[bias["outcome"] == outcome].sort_values("n_accounts")
        segs = b.drop_duplicates("segment")[["segment", "n_accounts"]]
        y = {s: i for i, s in enumerate(segs["segment"])}
        for pooling in POOLING_TYPES:
            d = b[b["model"] == pooling]
            ax.plot(d["error"] * 100, d["segment"].map(y) + offsets[pooling] * 0.8, "o",
                    color=MODEL_COLORS[pooling], ms=6)
        ax.axvline(0, color=COLORS["muted"], lw=1)
        ax.set_yticks(range(len(segs)), [f"{s} (n={n})" for s, n in zip(segs["segment"], segs["n_accounts"])],
                      fontsize=8)
        ax.set_xlabel("Average (estimate - truth) across datasets, percentage points")
        ax.set_title(f"Systematic bias by segment: {outcome} (smallest at bottom)", loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"Simulation study over {n_seeds} simulated datasets "
        "(faint dots: one dataset each; bars: average; small = under 100 accounts)",
        x=0.01, ha="left", fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--vary-truth", action="store_true")
    parser.add_argument("--summarise-only", action="store_true")
    args = parser.parse_args()
    tag = "_vary_truth" if args.vary_truth else ""

    if not args.summarise_only:
        run(args.seeds, args.vary_truth, tag)

    set_style()
    per_seed, summary, h2h, ll_h2h, ll_mean, bias = summarise(tag)
    plot_study(per_seed, bias, per_seed["seed"].nunique(), FIG_DIR / f"simulation_study{tag}.png")
    pd.set_option("display.width", 160)
    print(summary.round(4).to_string(index=False))
    print(h2h.round(2).to_string(index=False))
    print(ll_mean.round(4).to_string(index=False))
    print(ll_h2h.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
