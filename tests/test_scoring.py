from hkalpha.nlp.keyword import score_text
from hkalpha.nlp.llm import build_prompt, parse_response


def test_longest_phrase_wins():
    s, et, _ = score_text("POSITIVE PROFIT ALERT", "en")
    assert s > 0 and et == "earnings_positive"


def test_bilingual_core_events():
    assert score_text("PROFIT WARNING", "en")[0] < 0
    assert score_text("盈利警告", "zh")[0] < 0
    assert score_text("建議以協議安排方式私有化", "zh")[0] > 0
    assert score_text("根據一般授權配售新股", "zh")[0] < 0


def test_routine_is_zero_and_tagged():
    s, et, _ = score_text("MONTHLY RETURN OF EQUITY ISSUER ON MOVEMENTS IN SECURITIES", "en")
    assert s == 0 and et == "routine"


def test_llm_prompt_is_anonymised():
    row = {"title_en": "TENCENT HOLDINGS - SHARE BUY-BACK (Stock Code: 700)", "stock_name_en": "Tencent Holdings",
           "title_zh": "騰訊控股 股份回購", "stock_name_zh": "騰訊控股", "category_en": "", "category_zh": ""}
    p = build_prompt(row, "both")
    assert "tencent" not in p.lower() and "騰訊" not in p and "700" not in p


def test_parse_response_clips_and_validates():
    d = parse_response('```json\n{"event_type":"weird","sentiment":3,"surprise":-1,"confidence":0.4}\n```')
    assert d == {"event_type": "other", "sentiment": 1.0, "surprise": 0.0, "confidence": 0.4}
