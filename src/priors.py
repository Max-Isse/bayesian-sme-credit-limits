"""Priors for segment-level default and activation rates.

Each of the 12 segments is one industry x one size band. Its rate is built
on the log-odds scale from three pooled effects:

    logit(rate_s) = mu
                    + sigma_industry * a[industry of s]
                    + sigma_size     * b[size band of s]
                    + sigma_segment  * e[s]
    a, b, e ~ Normal(0, 1)

    mu             : the typical (portfolio-wide) log-odds rate
    sigma_industry : how much industries differ from each other
    sigma_size     : how much size bands differ from each other
    sigma_segment  : how much a segment differs from what its industry and
                     size band predict

So a tiny segment such as Hospitality | Medium borrows strength from the
other Hospitality segments, from the other Medium segments, and from the
portfolio as a whole. The data decide how much, by estimating the sigmas.

Writing each effect as sigma * (standard normal) is the "non-centred" form.
It is the same model as effect ~ Normal(0, sigma), but it is much easier for
the sampler when groups have little data (see model.py).
"""

import numpy as np
import pymc as pm
from scipy.special import logit

PRIORS = {
    # 12-month default rate among activated accounts.
    # mu centred on 5%; sd 0.75 means the portfolio-wide rate is 95% likely
    # to sit between about 1.2% and 19%.
    # The three spreads were split so that, together, they imply roughly the
    # same range of segment rates as a single HalfNormal(0.5) spread (checked
    # in prior_checks.py). Industry and size are expected to be the main
    # drivers, so the leftover segment-specific spread gets a smaller prior.
    "default": {
        "mu_mean": float(logit(0.05)),
        "mu_sd": 0.75,
        "sigma_industry_sd": 0.4,
        "sigma_size_sd": 0.4,
        "sigma_segment_sd": 0.25,
    },
    # 60-day activation rate. Less is known in advance, so mu is wider:
    # centred on 60%, portfolio-wide rate 95% likely between about 17% and 91%.
    "activation": {
        "mu_mean": float(logit(0.60)),
        "mu_sd": 1.0,
        "sigma_industry_sd": 0.4,
        "sigma_size_sd": 0.4,
        "sigma_segment_sd": 0.25,
    },
}

# No-pooling baseline: every segment gets its own independent, wider prior
# on the log-odds scale and shares nothing with the others.
NO_POOLING_SD = 1.5


def add_segment_rates(name, prior, industry_idx, size_idx):
    """Add the hierarchical (crossed-effects) segment rates to the current model.

    industry_idx and size_idx give, for each segment, the position of its
    industry and size band in the model's "industry" and "size_band" coords.
    """
    mu = pm.Normal(f"{name}_mu", mu=prior["mu_mean"], sigma=prior["mu_sd"])
    sigma_industry = pm.HalfNormal(f"{name}_sigma_industry", sigma=prior["sigma_industry_sd"])
    sigma_size = pm.HalfNormal(f"{name}_sigma_size", sigma=prior["sigma_size_sd"])
    sigma_segment = pm.HalfNormal(f"{name}_sigma_segment", sigma=prior["sigma_segment_sd"])

    a = pm.Normal(f"{name}_industry_z", 0.0, 1.0, dims="industry")
    b = pm.Normal(f"{name}_size_z", 0.0, 1.0, dims="size_band")
    e = pm.Normal(f"{name}_segment_z", 0.0, 1.0, dims="segment")

    log_odds = (
        mu
        + sigma_industry * a[industry_idx]
        + sigma_size * b[size_idx]
        + sigma_segment * e
    )
    return pm.Deterministic(f"{name}_rate", pm.math.invlogit(log_odds), dims="segment")


def sample_segment_rates_numpy(prior, n_draws, rng, industry_idx, size_idx):
    """Draw all segments' rates from the hierarchical prior with plain numpy.

    Returns an array of shape (n_draws, n_segments).
    """
    n_ind = industry_idx.max() + 1
    n_size = size_idx.max() + 1
    n_seg = len(industry_idx)

    def effects(sd, k):
        sigma = np.abs(rng.normal(0.0, sd, size=(n_draws, 1)))
        return sigma * rng.normal(size=(n_draws, k))

    mu = rng.normal(prior["mu_mean"], prior["mu_sd"], size=(n_draws, 1))
    log_odds = (
        mu
        + effects(prior["sigma_industry_sd"], n_ind)[:, industry_idx]
        + effects(prior["sigma_size_sd"], n_size)[:, size_idx]
        + effects(prior["sigma_segment_sd"], n_seg)
    )
    return 1.0 / (1.0 + np.exp(-log_odds))


def sample_flat_rates_numpy(mean, sd, n, rng):
    """Draw rates from a single Normal(mean, sd) prior on the log-odds scale."""
    return 1.0 / (1.0 + np.exp(-rng.normal(mean, sd, size=n)))
