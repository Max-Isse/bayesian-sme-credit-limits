"""Stage 4: validate the three models.

1. Parameter recovery   - do the estimates find the true segment rates?
2. Posterior predictive - can each fitted model reproduce the data it saw?
3. Calibration          - do predicted default rates match what happens on
                          a fresh, held-out portfolio?
4. Prior sensitivity    - do the hierarchical estimates change much if the
                          default priors change?

Run after fit_models.py:  python src/validate.py
Writes figures/validation_*.png and reports/validation_*.csv.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from scipy import stats
from scipy.special import logit

from data_generation import simulate_portfolio
from model import POOLING_TYPES, build_model, fit, prepare_data, segment_rate_summary
from plotting import COLORS, set_style
from priors import PRIORS

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = ROOT / "figures"

MODEL_LABELS = {"complete": "Complete pooling", "none": "No pooling", "partial": "Hierarchical"}
MODEL_COLORS = {
    "complete": COLORS["complete_pooling"],
    "none": COLORS["no_pooling"],
    "partial": COLORS["hierarchical"],
}
OUTCOMES = {"default": "12-month default rate (given activation)", "activation": "60-day activation rate"}
INTERVAL = 0.90
SMALL_SEGMENT = 100  # segments with fewer accounts than this count as "small"

# Hold-out portfolio: same true rates, a different seed, and observed long
# enough (snapshot at month 24) that every account has a full 12 months.
# It is 3x the size of the training data so that noise in the hold-out
# itself does not drown out differences between the models.
HOLDOUT = {"seed": 2025, "snapshot_month": 24, "size_multiplier": 3}

# Alternative default priors for the sensitivity check. Each changes one
# thing relative to the chosen priors.
SENSITIVITY_VARIANTS = {
    "Chosen priors": {},
    "Centred on 2%": {"mu_mean": float(logit(0.02))},
    "Centred on 10%": {"mu_mean": float(logit(0.10))},
    "Spreads doubled": {"sigma_industry_sd": 0.8, "sigma_size_sd": 0.8, "sigma_segment_sd": 0.5},
    "Spreads halved": {"sigma_industry_sd": 0.2, "sigma_size_sd": 0.2, "sigma_segment_sd": 0.125},
}
VARIANT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def load_fits():
    return {p: xr.open_datatree(RESULTS_DIR / f"idata_{p}.nc") for p in POOLING_TYPES}


# 1. Parameter recovery ------------------------------------------------------


def recovery_table(fits, truth):
    rows = []
    for pooling, idata in fits.items():
        for outcome in OUTCOMES:
            est = segment_rate_summary(idata, outcome, prob=INTERVAL)
            est = est.merge(truth[["segment", "n_accounts", f"true_{outcome}_rate"]], on="segment")
            est = est.rename(columns={f"true_{outcome}_rate": "true"})
            est["model"] = pooling
            est["outcome"] = outcome
            rows.append(est)
    table = pd.concat(rows, ignore_index=True)
    table["abs_error"] = (table["mean"] - table["true"]).abs()
    table["covered"] = (table["true"] >= table["lower"]) & (table["true"] <= table["upper"])
    table["interval_width"] = table["upper"] - table["lower"]
    return table


def recovery_summary(table):
    table = table.assign(size_group=np.where(table["n_accounts"] < SMALL_SEGMENT, "small", "large"))
    return (
        table.groupby(["outcome", "model", "size_group"])
        .agg(
            segments=("segment", "size"),
            mean_abs_error=("abs_error", "mean"),
            coverage=("covered", "mean"),
            mean_interval_width=("interval_width", "mean"),
        )
        .reset_index()
    )


def plot_recovery(table, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
    order = (
        table[["segment", "n_accounts"]].drop_duplicates().sort_values("n_accounts", ascending=False)
    )
    y_pos = {s: i for i, s in enumerate(order["segment"])}
    offsets = {"complete": -0.25, "none": 0.0, "partial": 0.25}

    for ax, outcome in zip(axes, OUTCOMES):
        sub = table[table["outcome"] == outcome]
        for pooling in POOLING_TYPES:
            m = sub[sub["model"] == pooling]
            y = m["segment"].map(y_pos) + offsets[pooling]
            ax.hlines(y, m["lower"], m["upper"], color=MODEL_COLORS[pooling], lw=2)
            ax.plot(m["mean"], y, "o", color=MODEL_COLORS[pooling], ms=5, label=MODEL_LABELS[pooling])
        truth = sub.drop_duplicates("segment")
        ax.plot(truth["true"], truth["segment"].map(y_pos), "|", color=COLORS["truth"],
                ms=16, mew=2.5, label="True rate")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"{s}  (n={n})" for s, n in zip(order["segment"], order["n_accounts"])])
        ax.set_xlabel(OUTCOMES[outcome])
        ax.grid(axis="y", visible=False)
        for i in range(len(order)):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="#f0efec", lw=0, zorder=0)
        ax.set_ylim(len(order) - 0.5, -0.5)
    axes[0].set_title("Default", loc="left")
    axes[1].set_title("Activation", loc="left")
    axes[1].set_yticklabels([])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Estimated vs true segment rates: posterior means with {INTERVAL:.0%} credible intervals"
        " (smallest segments at the bottom)",
        x=0.01, ha="left", fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def plot_error_by_size(table, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, outcome in zip(axes, OUTCOMES):
        sub = table[table["outcome"] == outcome]
        # Small sideways nudges stop the three models' dots hiding each other.
        for pooling, nudge in zip(POOLING_TYPES, [0.94, 1.0, 1.06]):
            m = sub[sub["model"] == pooling]
            ax.plot(m["n_accounts"] * nudge, m["abs_error"] * 100, "o", color=MODEL_COLORS[pooling],
                    ms=7, label=MODEL_LABELS[pooling])
        ax.axvline(SMALL_SEGMENT, color=COLORS["muted"], lw=1, ls="--")
        ax.set_xscale("log")
        ticks = [20, 50, 100, 200, 500]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Accounts in segment (log scale)")
        ax.set_ylabel("|estimate - truth| (percentage points)")
        ax.set_title("Default" if outcome == "default" else "Activation", loc="left")
    axes[0].legend()
    fig.suptitle("Estimation error by segment size", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# 2. Posterior predictive checks --------------------------------------------


def posterior_predictive(fits, data):
    """Replicated datasets from each fitted model, summarised per segment."""
    n_seg = len(data["segments"])
    out = {}
    for pooling, idata in fits.items():
        with build_model(data, pooling):
            pp = pm.sample_posterior_predictive(
                idata, var_names=["defaulted", "activated"], random_seed=5, progressbar=False
            )
        rep_def_groups = pp["posterior_predictive"]["defaulted"].values.reshape(-1, len(data["def_n"]))
        rep_def = np.zeros((rep_def_groups.shape[0], n_seg))
        for g, s in enumerate(data["def_seg_idx"]):
            rep_def[:, s] += rep_def_groups[:, g]
        rep_act = pp["posterior_predictive"]["activated"].values.reshape(-1, n_seg)
        out[pooling] = {"defaults": rep_def, "activated": rep_act}
    return out


def ppc_statistics(defaults, activated, data):
    """Summary statistics of one dataset (or of each replicated dataset).

    defaults and activated are per-segment counts, shape (..., n_segments).
    """
    seg_n_act = np.bincount(data["def_seg_idx"], weights=data["def_n"], minlength=len(data["segments"]))
    return {
        "Total visible defaults": defaults.sum(axis=-1),
        "Spread of segment default rates (SD)": (defaults / seg_n_act).std(axis=-1),
        "Spread of segment activation rates (SD)": (activated / data["act_n"]).std(axis=-1),
    }


def plot_ppc_statistics(reps, data, path):
    obs_def = np.bincount(data["def_seg_idx"], weights=data["def_k"], minlength=len(data["segments"]))
    observed = ppc_statistics(obs_def, data["act_k"], data)
    rows = []
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    for r, pooling in enumerate(POOLING_TYPES):
        replicated = ppc_statistics(reps[pooling]["defaults"], reps[pooling]["activated"], data)
        for c, (stat, values) in enumerate(replicated.items()):
            ax = axes[r, c]
            ax.hist(values, bins=40, color=MODEL_COLORS[pooling], edgecolor="#fcfcfb", linewidth=0.6)
            ax.axvline(observed[stat], color=COLORS["observed"], lw=2)
            p_value = np.mean(values >= observed[stat])
            ax.set_title(f"{stat}\nP(replicated >= observed) = {p_value:.2f}", fontsize=9, loc="left")
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(MODEL_LABELS[pooling], fontsize=10, color="#0b0b0b")
            rows.append({"model": pooling, "statistic": stat, "observed": observed[stat],
                         "replicated_median": np.median(values), "p_value": p_value})
    fig.suptitle("Posterior predictive checks: replicated data (bars) vs observed data (red line)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_ppc_segments(reps, data, path):
    """Hierarchical model: each segment's observed counts vs its predictive interval."""
    seg = data["segments"]["segment"].to_numpy()
    obs = {
        "defaults": np.bincount(data["def_seg_idx"], weights=data["def_k"], minlength=len(seg)),
        "activated": data["act_k"],
    }
    lo_q, hi_q = (1 - INTERVAL) / 2, 1 - (1 - INTERVAL) / 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, key, title in zip(axes, ["defaults", "activated"], ["Visible defaults", "Activations"]):
        rep = reps["partial"][key]
        lo, hi = np.quantile(rep, lo_q, axis=0), np.quantile(rep, hi_q, axis=0)
        y = np.arange(len(seg))
        ax.hlines(y, lo, hi, color=COLORS["hierarchical"], lw=6, alpha=0.35,
                  label=f"{INTERVAL:.0%} predictive interval")
        inside = (obs[key] >= lo) & (obs[key] <= hi)
        ax.plot(obs[key], y, "o", color=COLORS["truth"], ms=5, label="Observed")
        ax.set_yticks(y)
        ax.set_yticklabels(seg if key == "defaults" else [])
        ax.set_ylim(len(seg) - 0.5, -0.5)
        ax.set_title(f"{title}: {inside.sum()} of {len(seg)} segments inside interval", loc="left")
        ax.set_xlabel("Count")
        ax.grid(axis="y", visible=False)
    axes[1].legend(loc="lower right")
    fig.suptitle("Hierarchical model: posterior predictive check by segment", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# 3. Calibration on hold-out data ---------------------------------------------


def holdout_data():
    observed, _, _ = simulate_portfolio(**HOLDOUT)
    observed["activated_60d"] = observed["activated_60d"].astype(int)
    return observed


def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration(fits, holdout, small_segments):
    """Compare each segment's predicted rate with its hold-out rate.

    Predictions use the posterior mean rate. Default is checked on activated
    accounts only, since the default rate is P(default | activated).
    Scores are reported for all accounts and for the small segments alone,
    because the large segments dominate any portfolio-wide score.
    """
    seg_rows, metric_rows = [], []
    activated = holdout[holdout["activated_60d"] == 1]
    for pooling, idata in fits.items():
        for outcome, df, y_col in [
            ("default", activated, "defaulted"),
            ("activation", holdout, "activated_60d"),
        ]:
            pred = segment_rate_summary(idata, outcome).set_index("segment")["mean"]
            for scope, d in [("all", df), ("small", df[df["segment"].isin(small_segments)])]:
                p = d["segment"].map(pred).to_numpy()
                y = d[y_col].to_numpy()
                metric_rows.append({
                    "model": pooling, "outcome": outcome, "segments": scope, "accounts": len(d),
                    "log_loss": log_loss(y, p),
                    "brier": float(np.mean((p - y) ** 2)),
                    "predicted_total": float(p.sum()),
                    "observed_total": int(y.sum()),
                })
            obs = df.groupby("segment")[y_col].agg(["sum", "size"])
            for s, row in obs.iterrows():
                # Jeffreys interval: a standard interval for a binomial proportion.
                lo, hi = stats.beta.ppf([0.05, 0.95], row["sum"] + 0.5, row["size"] - row["sum"] + 0.5)
                seg_rows.append({
                    "model": pooling, "outcome": outcome, "segment": s,
                    "predicted": pred[s], "observed": row["sum"] / row["size"],
                    "obs_lower": lo, "obs_upper": hi, "n": row["size"],
                })
    return pd.DataFrame(seg_rows), pd.DataFrame(metric_rows)


def plot_calibration(seg_cal, metrics, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    limits = {"default": (0, 0.2), "activation": (0.4, 0.95)}
    for r, outcome in enumerate(OUTCOMES):
        for c, pooling in enumerate(POOLING_TYPES):
            ax = axes[r, c]
            d = seg_cal[(seg_cal["outcome"] == outcome) & (seg_cal["model"] == pooling)]
            lo, hi = limits[outcome]
            ax.plot([lo, hi], [lo, hi], color=COLORS["muted"], lw=1, ls="--", label="Perfect calibration")
            ax.vlines(d["predicted"], d["obs_lower"], d["obs_upper"], color=MODEL_COLORS[pooling], lw=1.5, alpha=0.6)
            ax.plot(d["predicted"], d["observed"], "o", color=MODEL_COLORS[pooling], ms=6)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            m = metrics[(metrics["outcome"] == outcome) & (metrics["model"] == pooling)
                        & (metrics["segments"] == "all")].iloc[0]
            ax.set_title(f"{MODEL_LABELS[pooling]}: {outcome}\nlog loss {m['log_loss']:.4f}, "
                         f"Brier {m['brier']:.4f}", loc="left", fontsize=10)
            ax.set_xlabel("Predicted rate (posterior mean)")
            if c == 0:
                ax.set_ylabel("Hold-out observed rate (90% interval)")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Calibration on a held-out portfolio (one point per segment)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# 4. Prior sensitivity --------------------------------------------------------


def prior_sensitivity(data, baseline_idata):
    rows = []
    for i, (label, changes) in enumerate(SENSITIVITY_VARIANTS.items()):
        if changes:
            priors = {**PRIORS, "default": {**PRIORS["default"], **changes}}
            idata = fit(build_model(data, "partial", priors=priors))
        else:
            idata = baseline_idata
        est = segment_rate_summary(idata, "default", prob=INTERVAL)
        est["variant"] = label
        rows.append(est)
    return pd.concat(rows, ignore_index=True)


def plot_sensitivity(sens, truth, path):
    order = truth.sort_values("n_accounts", ascending=False)
    y_pos = {s: i for i, s in enumerate(order["segment"])}
    variants = list(SENSITIVITY_VARIANTS)
    offsets = np.linspace(-0.3, 0.3, len(variants))
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for off, label, color in zip(offsets, variants, VARIANT_COLORS):
        d = sens[sens["variant"] == label]
        y = d["segment"].map(y_pos) + off
        ax.hlines(y, d["lower"], d["upper"], color=color, lw=1.8)
        ax.plot(d["mean"], y, "o", color=color, ms=4, label=label)
    ax.plot(order["true_default_rate"], order["segment"].map(y_pos), "|", color=COLORS["truth"],
            ms=18, mew=2.5, label="True rate")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{s}  (n={n})" for s, n in zip(order["segment"], order["n_accounts"])])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    for i in range(len(order)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f0efec", lw=0, zorder=0)
    ax.set_xlabel("12-month default rate (given activation)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.08), ncol=6, fontsize=8)
    ax.set_title("Prior sensitivity: hierarchical default estimates under alternative priors",
                 loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    set_style()
    REPORTS_DIR.mkdir(exist_ok=True)
    accounts = pd.read_csv(ROOT / "data" / "accounts.csv")
    truth = pd.read_csv(ROOT / "data" / "true_segment_rates.csv")
    data = prepare_data(accounts)
    fits = load_fits()

    rec = recovery_table(fits, truth)
    rec_summary = recovery_summary(rec)
    rec.to_csv(REPORTS_DIR / "validation_recovery_by_segment.csv", index=False)
    rec_summary.to_csv(REPORTS_DIR / "validation_recovery_summary.csv", index=False)
    plot_recovery(rec, FIG_DIR / "validation_recovery.png")
    plot_error_by_size(rec, FIG_DIR / "validation_error_by_size.png")
    print("Parameter recovery (90% intervals):")
    print(rec_summary.round(4).to_string(index=False))

    reps = posterior_predictive(fits, data)
    ppc = plot_ppc_statistics(reps, data, FIG_DIR / "validation_ppc_statistics.png")
    plot_ppc_segments(reps, data, FIG_DIR / "validation_ppc_segments.png")
    ppc.to_csv(REPORTS_DIR / "validation_ppc.csv", index=False)
    print("\nPosterior predictive checks:")
    print(ppc.round(4).to_string(index=False))

    small = truth.loc[truth["n_accounts"] < SMALL_SEGMENT, "segment"]
    seg_cal, metrics = calibration(fits, holdout_data(), small)
    seg_cal.to_csv(REPORTS_DIR / "validation_calibration_by_segment.csv", index=False)
    metrics.to_csv(REPORTS_DIR / "validation_calibration_metrics.csv", index=False)
    plot_calibration(seg_cal, metrics, FIG_DIR / "validation_calibration.png")
    print("\nHold-out calibration:")
    print(metrics.round(5).to_string(index=False))

    sens = prior_sensitivity(data, fits["partial"])
    sens.to_csv(REPORTS_DIR / "validation_prior_sensitivity.csv", index=False)
    plot_sensitivity(sens, truth, FIG_DIR / "validation_prior_sensitivity.png")
    base = sens[sens["variant"] == "Chosen priors"].set_index("segment")["mean"]
    shift = sens.assign(shift=sens["mean"] - sens["segment"].map(base))
    shift = shift.merge(truth[["segment", "n_accounts"]], on="segment")
    print("\nPrior sensitivity: largest change in a segment's posterior mean default rate (pp):")
    shift = shift[shift["variant"] != "Chosen priors"]
    print(
        shift.assign(abs_shift_pp=shift["shift"].abs() * 100)
        .groupby("variant")
        .apply(lambda d: d.loc[d["abs_shift_pp"].idxmax(), ["segment", "n_accounts", "abs_shift_pp"]],
               include_groups=False)
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
