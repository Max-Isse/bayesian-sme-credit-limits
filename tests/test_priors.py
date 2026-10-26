import numpy as np
import pymc as pm

from data_generation import simulate_portfolio
from prior_checks import build_prior_model
from priors import PRIORS, sample_segment_rates_numpy


def test_default_prior_puts_most_mass_on_plausible_rates():
    rng = np.random.default_rng(0)
    p = PRIORS["default"]
    rates = sample_segment_rates_numpy(p["mu_mean"], p["mu_sd"], p["sigma_sd"], 100_000, rng)
    assert np.mean(rates <= 0.20) > 0.90
    assert np.mean(rates > 0.50) < 0.01


def test_prior_model_builds_and_samples():
    observed, _, _ = simulate_portfolio()
    model, segments = build_prior_model(observed)
    assert len(segments) == 12
    with model:
        prior = pm.sample_prior_predictive(draws=50, random_seed=1)
    rates = prior.prior["default_rate"].values
    assert rates.shape[-1] == 12
    assert ((rates > 0) & (rates < 1)).all()
