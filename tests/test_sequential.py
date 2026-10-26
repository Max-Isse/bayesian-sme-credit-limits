import pandas as pd

from data_generation import simulate_portfolio
from sequential import MONTHS, snapshot


def test_final_snapshot_is_the_stage_1_data():
    final = snapshot(12)
    full, _, _ = simulate_portfolio()
    pd.testing.assert_frame_equal(final, full)


def test_snapshots_only_ever_add_information():
    previous = None
    for month in MONTHS:
        s = snapshot(month).set_index("account_id")
        assert s["cohort_month"].max() <= month
        if previous is not None:
            # Every account seen last month is still there, one month older,
            # and a default once visible stays visible.
            assert previous.index.isin(s.index).all()
            same = s.loc[previous.index]
            assert (same["months_on_book"] == previous["months_on_book"] + 1).all()
            assert (same["defaulted"] >= previous["defaulted"]).all()
        previous = s


def test_nothing_is_known_about_activation_at_month_one():
    assert snapshot(1)["activated_60d"].isna().all()
