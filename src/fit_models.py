"""Stage 3: fit the three models and check that the sampler converged.

Run:  python src/fit_models.py
Writes results/idata_<pooling>.nc, results/convergence.csv and
figures/convergence_trace.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import POOLING_TYPES, build_model, convergence_table, fit, n_divergences, prepare_data
from plotting import set_style

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

# Parameters worth eyeballing: the top level of the hierarchy is where
# sampling problems usually show up first.
TRACE_PARAMS = ["default_mu", "default_sigma_industry", "default_sigma_size", "default_sigma_segment"]
CHAIN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def plot_traces(idata, path):
    """Left: each chain's histogram. Right: each chain's draws in order.

    Healthy chains overlap on the left and look like flat, well-mixed
    'hairy caterpillars' on the right, with no trends or stuck stretches.
    """
    post = idata["posterior"]
    fig, axes = plt.subplots(len(TRACE_PARAMS), 2, figsize=(11, 2.2 * len(TRACE_PARAMS)),
                             gridspec_kw={"width_ratios": [1, 2.2]})
    for row, name in zip(axes, TRACE_PARAMS):
        values = post[name].values  # (chain, draw)
        bins = np.linspace(values.min(), values.max(), 40)
        for c in range(values.shape[0]):
            color = CHAIN_COLORS[c % len(CHAIN_COLORS)]
            row[0].hist(values[c], bins=bins, histtype="step", color=color, lw=1.5, label=f"Chain {c + 1}")
            row[1].plot(values[c], color=color, lw=0.5, alpha=0.8)
        row[0].set_title(name, loc="left")
        row[0].set_yticks([])
        row[1].set_xlabel("Draw")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Hierarchical default model: chains for the top-level parameters", x=0.01,
                 ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    set_style()
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    accounts = pd.read_csv(ROOT / "data" / "accounts.csv")
    data = prepare_data(accounts)

    rows = []
    for pooling in POOLING_TYPES:
        idata = fit(build_model(data, pooling))
        idata.to_netcdf(RESULTS_DIR / f"idata_{pooling}.nc")
        table = convergence_table(idata)
        rows.append(
            {
                "model": pooling,
                "n_parameters": len(table),
                "max_r_hat": table["r_hat"].max(),
                "min_ess_bulk": table["ess_bulk"].min(),
                "min_ess_tail": table["ess_tail"].min(),
                "divergences": n_divergences(idata),
            }
        )
        if pooling == "partial":
            plot_traces(idata, FIG_DIR / "convergence_trace.png")
            table.to_csv(RESULTS_DIR / "convergence_partial_by_parameter.csv")

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "convergence.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
