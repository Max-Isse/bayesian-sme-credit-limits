import numpy as np
import pymc as pm

from data_generation import simulate_portfolio
from model import build_model, prepare_data, segment_frame
from priors import PRIORS, sample_segment_rates_numpy


def test_default_prior_puts_most_mass_on_plausible_rates():
    rng = np.random.default_rng(0)
    seg = segment_frame()
    rates = sample_segment_rates_numpy(
        PRIORS["default"], 10_000, rng, seg["industry_idx"].to_numpy(), seg["size_idx"].to_numpy()
    )
    assert rates.shape == (10_000, 12)
    assert np.mean(rates <= 0.20) > 0.90
    assert np.mean(rates > 0.50) < 0.01


def test_prior_predictive_runs_on_the_fitted_model():
    observed, _, _ = simulate_portfolio()
    with build_model(prepare_data(observed), pooling="partial"):
        prior = pm.sample_prior_predictive(draws=50, random_seed=1)
    rates = prior.prior["default_rate"].values
    assert rates.shape[-1] == 12
    assert ((rates > 0) & (rates < 1)).all()
