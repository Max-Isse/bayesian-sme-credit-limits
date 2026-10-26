"""Parameter recovery: fit the hierarchical model to data simulated with known
rates and check it finds them.

Uses a fresh seed (not the training data) with every account observed for a
full year, so the test checks the model rather than one particular draw.
The pass criteria were fixed before the test was first run.
"""

import numpy as np

from data_generation import simulate_portfolio
from model import build_model, fit, n_divergences, prepare_data, segment_rate_summary


def test_hierarchical_model_recovers_true_segment_rates():
    observed, _, truth = simulate_portfolio(seed=101, snapshot_month=24, size_multiplier=2)
    idata = fit(build_model(prepare_data(observed), "partial"), draws=1000, tune=1000, chains=4)
    assert n_divergences(idata) == 0

    for outcome, max_mean_abs_error in [("default", 0.02), ("activation", 0.04)]:
        est = segment_rate_summary(idata, outcome, prob=0.90).merge(
            truth[["segment", f"true_{outcome}_rate"]], on="segment"
        )
        true = est[f"true_{outcome}_rate"]
        covered = ((true >= est["lower"]) & (true <= est["upper"])).sum()
        mean_abs_error = np.mean(np.abs(est["mean"] - true))
        assert covered >= 9, f"{outcome}: only {covered}/12 true rates inside 90% intervals"
        assert mean_abs_error < max_mean_abs_error, f"{outcome}: mean abs error {mean_abs_error:.3f}"
