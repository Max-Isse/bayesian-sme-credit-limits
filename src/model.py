"""Segment-level models for activation and default, with three pooling choices.

    "complete": one rate shared by every segment (ignores segment differences)
    "none":     a separate, independent rate per segment (no sharing at all)
    "partial":  the hierarchical model in priors.py; segments share strength
                through industry, size band and the portfolio average

Likelihoods
-----------
Activation: in each segment, the accounts whose 60-day window has closed
    activate with the segment's activation rate:
        activated_s ~ Binomial(n_known_s, activation_rate_s)

Default: an activated account observed for m months (m <= 12) has had m/12
    of its 12-month default window. Assuming default timing is spread evenly
    over the year, the chance its default is already visible is
        default_rate_s * m / 12
    Accounts with the same segment and months on book share this probability,
    so we group them and use one Binomial per group (same answer as one
    Bernoulli per account, but faster):
        defaults_{s,m} ~ Binomial(n_{s,m}, default_rate_s * m / 12)
"""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from data_generation import INDUSTRIES, SIZE_BANDS, segment_table
from priors import NO_POOLING_SD, PRIORS, add_segment_rates

POOLING_TYPES = ["complete", "none", "partial"]


def segment_frame():
    """The 12 segments in a fixed order, with industry and size positions."""
    seg = segment_table()[["segment", "industry", "size_band"]].copy()
    seg["industry_idx"] = seg["industry"].map({v: i for i, v in enumerate(INDUSTRIES)})
    seg["size_idx"] = seg["size_band"].map({v: i for i, v in enumerate(SIZE_BANDS)})
    return seg.reset_index(drop=True)


def prepare_data(accounts):
    """Turn account rows into the counts each likelihood needs.

    Accounts whose 60-day activation window is still open are left out of
    both models: we don't yet know whether they activated, and only
    activated accounts can default.
    """
    seg = segment_frame()
    seg_pos = {s: i for i, s in enumerate(seg["segment"])}

    known = accounts[accounts["activated_60d"].notna()].copy()
    known["activated_60d"] = known["activated_60d"].astype(int)
    act = known.groupby("segment")["activated_60d"].agg(["size", "sum"]).reindex(seg["segment"])
    act = act.fillna(0).astype(int)

    activated = known[known["activated_60d"] == 1]
    dflt = (
        activated.groupby(["segment", "months_on_book"])["defaulted"]
        .agg(["size", "sum"])
        .reset_index()
    )
    return {
        "segments": seg,
        "act_n": act["size"].to_numpy(),
        "act_k": act["sum"].to_numpy(),
        "def_seg_idx": dflt["segment"].map(seg_pos).to_numpy(),
        "def_exposure": (np.minimum(dflt["months_on_book"], 12) / 12).to_numpy(),
        "def_n": dflt["size"].to_numpy(),
        "def_k": dflt["sum"].to_numpy(),
    }


def _segment_rates(name, pooling, seg, priors):
    prior = priors[name]
    if pooling == "partial":
        return add_segment_rates(
            name, prior, seg["industry_idx"].to_numpy(), seg["size_idx"].to_numpy()
        )
    if pooling == "none":
        log_odds = pm.Normal(
            f"{name}_log_odds", mu=prior["mu_mean"], sigma=NO_POOLING_SD, dims="segment"
        )
        return pm.Deterministic(f"{name}_rate", pm.math.invlogit(log_odds), dims="segment")
    if pooling == "complete":
        log_odds = pm.Normal(f"{name}_log_odds", mu=prior["mu_mean"], sigma=prior["mu_sd"])
        rate = pm.math.invlogit(log_odds) * np.ones(len(seg))
        return pm.Deterministic(f"{name}_rate", rate, dims="segment")
    raise ValueError(f"pooling must be one of {POOLING_TYPES}")


def build_model(data, pooling="partial", priors=PRIORS):
    """Activation and default models for one pooling choice.

    The two outcomes share no parameters, so fitting them in one PyMC model
    gives exactly the same answer as fitting them separately.
    `priors` can be swapped out to test how sensitive the results are to them.
    """
    seg = data["segments"]
    coords = {"segment": seg["segment"].tolist(), "industry": INDUSTRIES, "size_band": SIZE_BANDS}
    with pm.Model(coords=coords) as model:
        act_rate = _segment_rates("activation", pooling, seg, priors)
        def_rate = _segment_rates("default", pooling, seg, priors)

        pm.Binomial("activated", n=data["act_n"], p=act_rate, observed=data["act_k"], dims="segment")
        pm.Binomial(
            "defaulted",
            n=data["def_n"],
            p=def_rate[data["def_seg_idx"]] * data["def_exposure"],
            observed=data["def_k"],
        )
    return model


def fit(model, draws=1000, tune=1000, chains=4, target_accept=0.95, random_seed=42):
    """Sample the posterior with NUTS.

    target_accept=0.95 (default 0.8) makes the sampler take smaller steps.
    That is slower but avoids divergences in the narrow regions that
    hierarchical models create when a group spread (sigma) is close to zero.
    """
    with model:
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
        )


def convergence_table(idata):
    """R-hat, effective sample sizes and divergences for every sampled parameter."""
    summary = pd.DataFrame(az.summary(idata, kind="diagnostics", round_to="none"))
    return summary[["r_hat", "ess_bulk", "ess_tail"]]


def n_divergences(idata):
    return int(idata["sample_stats"]["diverging"].sum().item())


def segment_rate_summary(idata, name, prob=0.90):
    """Posterior mean and central credible interval of each segment's rate."""
    draws = idata["posterior"][f"{name}_rate"].stack(sample=("chain", "draw"))
    lo, hi = (1 - prob) / 2, 1 - (1 - prob) / 2
    return pd.DataFrame(
        {
            "segment": draws["segment"].values,
            "mean": draws.mean("sample").values,
            "lower": draws.quantile(lo, dim="sample").values,
            "upper": draws.quantile(hi, dim="sample").values,
        }
    )
