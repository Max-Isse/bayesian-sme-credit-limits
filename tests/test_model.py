import numpy as np
import pytest

from data_generation import simulate_portfolio
from model import POOLING_TYPES, build_model, fit, n_divergences, prepare_data


@pytest.fixture(scope="module")
def observed():
    return simulate_portfolio()[0]


def test_prepare_data_counts_add_up(observed):
    data = prepare_data(observed)
    known = observed[observed["activated_60d"].notna()]
    activated = known[known["activated_60d"] == 1]
    assert data["act_n"].sum() == len(known)
    assert data["act_k"].sum() == len(activated)
    assert data["def_n"].sum() == len(activated)
    assert data["def_k"].sum() == activated["defaulted"].sum()
    assert ((data["def_exposure"] > 0) & (data["def_exposure"] <= 1)).all()


@pytest.mark.parametrize("pooling", POOLING_TYPES)
def test_each_model_builds_with_one_rate_per_segment(observed, pooling):
    model = build_model(prepare_data(observed), pooling)
    for name in ["default_rate", "activation_rate"]:
        assert model.named_vars_to_dims[name] == ("segment",)


def test_complete_pooling_gives_every_segment_the_same_rate(observed):
    model = build_model(prepare_data(observed), "complete")
    draws = model.compile_fn(model["default_rate"], inputs=model.value_vars, on_unused_input="ignore")(
        model.initial_point()
    )
    assert np.allclose(draws, draws[0])


def test_short_hierarchical_fit_runs_cleanly(observed):
    idata = fit(build_model(prepare_data(observed), "partial"), draws=200, tune=300, chains=2)
    rates = idata["posterior"]["default_rate"].values
    assert np.isfinite(rates).all()
    assert n_divergences(idata) == 0
