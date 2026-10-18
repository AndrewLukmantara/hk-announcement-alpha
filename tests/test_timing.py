import pandas as pd

from hkalpha.timing import assign_entry_dates

DAYS = pd.to_datetime(["2025-03-03", "2025-03-04", "2025-03-05", "2025-03-07"])  # 03-06 = holiday


def _entry(ts, lag=30):
    ann = pd.DataFrame({"datetime": [pd.Timestamp(ts)]})
    out = assign_entry_dates(ann, DAYS, lag)
    return out["entry_date"].iloc[0], out["session"].iloc[0]


def test_post_close_goes_to_next_day():
    assert _entry("2025-03-03 18:30") == (pd.Timestamp("2025-03-04"), "post_close")


def test_pre_open_and_lunch_trade_same_close():
    assert _entry("2025-03-04 08:00") == (pd.Timestamp("2025-03-04"), "pre_open")
    assert _entry("2025-03-04 12:30") == (pd.Timestamp("2025-03-04"), "lunch")


def test_late_afternoon_respects_lag():
    assert _entry("2025-03-04 15:20")[0] == pd.Timestamp("2025-03-04")
    assert _entry("2025-03-04 15:50")[0] == pd.Timestamp("2025-03-05")


def test_holidays_and_weekends_roll_forward():
    assert _entry("2025-03-05 19:00")[0] == pd.Timestamp("2025-03-07")
    assert _entry("2025-03-06 10:00") == (pd.Timestamp("2025-03-07"), "non_trading_day")


def test_after_last_day_is_nat():
    assert pd.isna(_entry("2025-03-07 17:00")[0])
