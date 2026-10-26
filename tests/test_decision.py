import numpy as np

from decision import choose_limit, expected_card_spend


def test_expected_card_spend_matches_simulation():
    rng = np.random.default_rng(0)
    mu, sigma, limit = np.log(1500), 0.6, 2000
    simulated = np.minimum(rng.lognormal(mu, sigma, 1_000_000), limit).mean()
    assert np.isclose(expected_card_spend(limit, mu, sigma), simulated, rtol=0.005)


def test_riskier_segments_get_lower_limits():
    pa = np.full(2000, 0.7)
    limits = [choose_limit(np.full(2000, p), pa, np.log(3500), 0.6)["limit"] for p in [0.02, 0.05, 0.08, 0.12]]
    assert limits == sorted(limits, reverse=True)
    assert limits[0] > limits[-1]


def test_more_uncertainty_never_raises_the_limit():
    rng = np.random.default_rng(1)
    pa = np.full(4000, 0.7)
    centre, z = np.log(0.05 / 0.95), rng.normal(size=4000)
    limits = []
    for spread in [0.0, 0.2, 0.4, 0.6]:
        pd_draws = 1 / (1 + np.exp(-(centre + spread * z)))
        limits.append(choose_limit(pd_draws, pa, np.log(3500), 0.6)["limit"])
    assert limits == sorted(limits, reverse=True)


def test_very_risky_segment_is_declined():
    choice = choose_limit(np.full(1000, 0.4), np.full(1000, 0.7), np.log(3500), 0.6)
    assert choice["limit"] == 0
