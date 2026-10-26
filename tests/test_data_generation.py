import numpy as np
import pandas as pd

from data_generation import TRUE_PARAMS, segment_table, simulate_portfolio


def test_same_seed_gives_identical_data():
    a, _, _ = simulate_portfolio()
    b, _, _ = simulate_portfolio()
    pd.testing.assert_frame_equal(a, b)


def test_segment_sizes_match_settings_and_are_unbalanced():
    observed, _, seg = simulate_portfolio()
    assert len(seg) == 12
    counts = observed["segment"].value_counts()
    for _, s in seg.iterrows():
        assert counts[s["segment"]] == s["n_accounts"]
    assert (counts < 30).sum() >= 2
    assert counts.max() >= 300


def test_true_rates_are_plausible():
    seg = segment_table()
    assert seg["true_default_rate"].between(0.01, 0.20).all()
    assert seg["true_activation_rate"].between(0.40, 0.95).all()


def test_later_cohorts_have_less_history():
    observed, _, _ = simulate_portfolio()
    mob = observed.groupby("cohort_month")["months_on_book"].unique()
    for cohort, values in mob.items():
        assert list(values) == [TRUE_PARAMS["n_cohorts"] - cohort + 1]


def test_only_activated_accounts_can_default():
    observed, truth, _ = simulate_portfolio()
    assert truth.loc[truth["activated_60d_true"] == 0, "default_12m_true"].sum() == 0
    known = observed["activated_60d"].notna()
    assert observed.loc[known & (observed["activated_60d"] == 0), "defaulted"].sum() == 0


def test_observed_defaults_respect_censoring():
    observed, truth, _ = simulate_portfolio()
    m = observed.merge(truth, on="account_id")
    # Every visible default is a true default, and it happened on or before the snapshot.
    seen = m[m["defaulted"] == 1]
    assert (seen["default_12m_true"] == 1).all()
    assert (seen["default_month"] <= seen["months_on_book"]).all()
    # Every true default that happened before the snapshot is visible.
    should_see = (m["default_12m_true"] == 1) & (m["default_month_true"] <= m["months_on_book"])
    assert (m.loc[should_see, "defaulted"] == 1).all()


def test_activation_unknown_only_inside_60_day_window():
    observed, _, _ = simulate_portfolio()
    assert (observed["activated_60d"].isna() == (observed["months_on_book"] < 2)).all()


def test_large_sample_matches_true_rates():
    # With many accounts and every cohort past 12 months, empirical rates
    # should match the true segment rates closely.
    observed, truth, seg = simulate_portfolio(seed=123, snapshot_month=24, size_multiplier=40)
    m = observed.merge(truth, on="account_id")
    act = m.groupby("segment")["activated_60d"].mean()
    dflt = m[m["activated_60d"] == 1].groupby("segment")["defaulted"].mean()
    s = seg.set_index("segment")
    assert np.allclose(act[s.index], s["true_activation_rate"], atol=0.03)
    assert np.allclose(dflt[s.index], s["true_default_rate"], atol=0.03)
