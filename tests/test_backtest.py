import numpy as np
import pandas as pd

from hkalpha.backtest import Costs, build_weights, run_backtest
from hkalpha.signal import decayed, event_matrix


def test_cost_per_side_includes_stamp_duty():
    c = Costs(commission_bps=0, slippage_bps=0)
    assert abs(c.per_side - (0.001 + 0.0000565 + 0.000027 + 0.0000015)) < 1e-12


def test_returns_are_earned_next_day_and_costs_charged():
    dates = pd.bdate_range("2025-01-01", periods=4)
    w = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0]}, index=dates)
    r = pd.DataFrame({"A": [0.5, 0.01, 0.02, 0.5]}, index=dates)
    c = Costs(commission_bps=0, slippage_bps=0, borrow_bps_annual=0)
    bt = run_backtest(w, r, c)
    assert bt["gross"].tolist() == [0.0, 0.01, 0.02, 0.0]  # day-0 and day-3 moves not captured
    assert np.isclose(bt["cost"].iloc[1], c.per_side) and np.isclose(bt["cost"].iloc[3], c.per_side)


def test_long_short_is_dollar_neutral():
    dates = pd.bdate_range("2025-01-01", periods=2)
    s = pd.DataFrame([[3, 2, 1, -1, -2, -3]] * 2, index=dates, columns=list("ABCDEF"), dtype=float)
    w = build_weights(s, s.notna(), "long_short", quantile=0.5, min_names=3)
    assert np.allclose(w.sum(axis=1), 0) and np.allclose(w.abs().sum(axis=1), 1)


def test_signal_has_no_lookahead():
    dates = pd.bdate_range("2025-01-01", periods=30)
    ev = pd.DataFrame({"entry_date": [dates[5], dates[20]], "stock_code": ["A", "A"], "v": [1.0, -1.0]})
    s1 = decayed(event_matrix(ev, "v", dates, ["A"]), 5)
    ev.loc[1, "v"] = 99.0  # change a future event
    s2 = decayed(event_matrix(ev, "v", dates, ["A"]), 5)
    assert s1.iloc[:20].equals(s2.iloc[:20])
    assert (s1.iloc[:5] == 0).all().all()
