"""Priors for segment-level default and activation rates.

Each segment's rate is modelled on the log-odds scale:

    logit(rate_s) = mu + sigma * z_s,    z_s ~ Normal(0, 1)

    mu    : the typical (portfolio-wide) log-odds rate
    sigma : how far segments typically sit from that typical rate
    z_s   : where segment s sits, in units of sigma

This is the same as logit(rate_s) ~ Normal(mu, sigma). Writing it via z_s
(the "non-centred" form) changes nothing about the model, but it helps the
sampler when some segments have little data. Stage 3 explains why.
"""

import numpy as np
import pymc as pm
from scipy.special import logit

PRIORS = {
    # 12-month default rate among activated accounts.
    # mu centred on 5%; sd 0.75 on the log-odds scale means the portfolio-wide
    # rate is 95% likely to sit between about 1.2% and 19%.
    # sigma ~ HalfNormal(0.5): median 0.34, 95% below 0.98. At the median, a
    # segment two sigmas from typical has about 2x (or 1/2x) the typical odds.
    "default": {"mu_mean": float(logit(0.05)), "mu_sd": 0.75, "sigma_sd": 0.5},
    # 60-day activation rate. Less is known in advance, so the prior is wider:
    # centred on 60%, portfolio-wide rate 95% likely between about 17% and 91%.
    "activation": {"mu_mean": float(logit(0.60)), "mu_sd": 1.0, "sigma_sd": 0.5},
}


def add_segment_rates(name, prior, dims="segment"):
    """Add the hierarchical segment-rate priors to the current PyMC model."""
    mu = pm.Normal(f"{name}_mu", mu=prior["mu_mean"], sigma=prior["mu_sd"])
    sigma = pm.HalfNormal(f"{name}_sigma", sigma=prior["sigma_sd"])
    z = pm.Normal(f"{name}_z", mu=0.0, sigma=1.0, dims=dims)
    return pm.Deterministic(f"{name}_rate", pm.math.invlogit(mu + sigma * z), dims=dims)


def sample_segment_rates_numpy(mu_mean, mu_sd, sigma_sd, n, rng):
    """Draw segment rates from the prior with plain numpy (for quick comparisons).

    sigma_sd=0 means no between-segment spread, i.e. a prior on a single rate.
    """
    mu = rng.normal(mu_mean, mu_sd, size=n)
    sigma = np.abs(rng.normal(0.0, sigma_sd, size=n)) if sigma_sd > 0 else 0.0
    return 1.0 / (1.0 + np.exp(-(mu + sigma * rng.normal(size=n))))
