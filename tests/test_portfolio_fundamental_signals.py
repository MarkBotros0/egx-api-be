"""
Portfolio P/E and dividend-yield signals, per market.

The EG half is a PIN written against the code BEFORE it was made per-market
(2026-10-16): every threshold boundary and every string, literally. If one of
these fails, EG output changed — that is a bug, not a test to update.
"""
from __future__ import annotations

import os

os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-gate")
os.environ.setdefault("DATABASE_URL", "postgresql://unused/unused")

import pytest  # noqa: E402

from app.core.markets import EG  # noqa: E402
from app.routers.portfolio_analysis import fundamental_signals  # noqa: E402

LOW = "A P/E under 3 on the EGX almost always means the earnings were one-off, or the share price has already collapsed. Check the last results before treating this as cheap."
CHEAP = "A low P/E can indicate value, but it can also mean the market expects earnings to fall. Always combine with trend and relative strength before acting."
DEAR = "A high P/E means the market is paying a lot for each EGP of current earnings. Only justified by strong, confirmed growth expectations."
SOLID = "Read this as evidence the company generates real cash and returns it, not as income: even this yield loses to the ~19% policy rate. It is a quality marker, not a reason to buy on its own."
EXTREME = "A yield this high is usually a one-off special dividend, or it only looks high because the share price collapsed. Confirm the payout recurs before treating it as a reason to hold."
LOSS = "The company isn't profitable right now, so earnings-based valuation doesn't apply — lean on trend, relative strength, and macro instead."

EG_PIN = [
    ({"pe_ratio": None}, []),
    ({"pe_ratio": 0.7}, [("pe_implausibly_low", "warning", "COMI P/E is 0.7 — implausibly low, not a bargain.", LOW, "pe_ratio")]),
    ({"pe_ratio": 2.99}, [("pe_implausibly_low", "warning", "COMI P/E is 3.0 — implausibly low, not a bargain.", LOW, "pe_ratio")]),
    ({"pe_ratio": 3.0}, [("pe_undervalued", "opportunity", "COMI P/E is 3.0 — cheap versus the EGX median of ~12.", CHEAP, "pe_ratio")]),
    ({"pe_ratio": 7.99}, [("pe_undervalued", "opportunity", "COMI P/E is 8.0 — cheap versus the EGX median of ~12.", CHEAP, "pe_ratio")]),
    ({"pe_ratio": 8.0}, []),
    ({"pe_ratio": 12.4}, []),
    ({"pe_ratio": 24.99}, []),
    ({"pe_ratio": 25.0}, [("pe_overvalued", "warning", "COMI P/E is 25.0 — expensive versus the EGX median of ~12.", DEAR, "pe_ratio")]),
    ({"pe_ratio": 60.0}, [("pe_overvalued", "warning", "COMI P/E is 60.0 — expensive versus the EGX median of ~12.", DEAR, "pe_ratio")]),
    ({"dividend_yield": 0.0}, []),
    ({"dividend_yield": 3.99}, []),
    ({"dividend_yield": 4.0}, [("dividend_yield_solid", "info", "COMI yields 4.0% — an above-median EGX payer.", SOLID, "dividend_yield")]),
    ({"dividend_yield": 14.99}, [("dividend_yield_solid", "info", "COMI yields 15.0% — an above-median EGX payer.", SOLID, "dividend_yield")]),
    ({"dividend_yield": 15.0}, [("dividend_yield_extreme", "warning", "COMI yields 15.0% — check before counting on it.", EXTREME, "dividend_yield")]),
    ({"dividend_yield": 42.55}, [("dividend_yield_extreme", "warning", "COMI yields 42.5% — check before counting on it.", EXTREME, "dividend_yield")]),
    ({"loss_making": True, "pe_ratio": None, "dividend_yield": 0.0}, [("pe_loss_making", "warning", "COMI is loss-making over the last twelve months.", LOSS, "pe_ratio")]),
]


def _flat(sigs):
    return [(s["type"], s["severity"], s["message"], s["explanation"], s["learn_concept"]) for s in sigs]


@pytest.mark.parametrize("info,expected", EG_PIN)
def test_eg_signals_are_pinned(info, expected):
    sigs = fundamental_signals("COMI", info, EG, 19.0)
    assert _flat(sigs) == expected
    assert all(s["symbol"] == "COMI" for s in sigs)


# ---------------------------------------------------------------------------
# US: the US boundaries and "the US median", no EGX/Egypt facts
# ---------------------------------------------------------------------------

from app.core.calibration import for_market  # noqa: E402
from app.core.markets import US  # noqa: E402

US_FUND = for_market(US).fundamentals
EG_FUND = for_market(EG).fundamentals


def _types(info, market=US, rfr=4.2):
    return [s["type"] for s in fundamental_signals("AAPL", info, market, rfr)]


def test_eg_thresholds_are_the_eg_bundle_boundaries():
    """The pin above holds BECAUSE the EG bundle's boundaries are 3/8/25 and 4/15."""
    assert [b[0] for b in EG_FUND.pe_bands[:4]] == [3.0, 8.0, 15.0, 25.0]
    assert [b[0] for b in EG_FUND.dy_bands[:4]] == [2.0, 4.0, 8.0, 15.0]


def test_us_pe_signals_use_the_us_boundaries():
    lo, cheap, dear = US_FUND.pe_bands[0][0], US_FUND.pe_bands[1][0], US_FUND.pe_bands[3][0]
    assert _types({"pe_ratio": lo - 0.01}) == []      # EG-only concept
    assert _types({"pe_ratio": lo}) == ["pe_undervalued"]
    assert _types({"pe_ratio": cheap - 0.01}) == ["pe_undervalued"]
    assert _types({"pe_ratio": cheap}) == []
    assert _types({"pe_ratio": US_FUND.pe_median}) == []
    assert _types({"pe_ratio": dear - 0.01}) == []
    assert _types({"pe_ratio": dear}) == ["pe_overvalued"]
    # an EG-"expensive" P/E of 25 is unremarkable in the US
    assert _types({"pe_ratio": 25.0}) == []


def test_us_dividend_signals_use_the_us_boundaries():
    solid, extreme = US_FUND.dy_bands[1][0], US_FUND.dy_bands[3][0]
    assert _types({"dividend_yield": solid - 0.01}) == []
    assert _types({"dividend_yield": solid}) == ["dividend_yield_solid"]
    assert _types({"dividend_yield": extreme - 0.01}) == ["dividend_yield_solid"]
    assert _types({"dividend_yield": extreme}) == []    # EG-only concept


def test_us_copy_names_the_us_median_and_its_currency():
    msg = fundamental_signals("AAPL", {"pe_ratio": 60.0}, US, 4.2)[0]
    assert f"the US median of ~{US_FUND.pe_median:.0f}" in msg["message"]
    assert "each USD of current earnings" in msg["explanation"]
    cheap = fundamental_signals("AAPL", {"pe_ratio": US_FUND.pe_bands[0][0]}, US, 4.2)[0]
    assert "the US median" in cheap["message"]


def test_us_dividend_copy_makes_no_cash_rate_claim_the_bundle_did_not_measure():
    assert US_FUND.dy_mid_under_risk_free is False
    s = fundamental_signals("AAPL", {"dividend_yield": 3.0}, US, 4.2)[0]
    assert "loses to" not in s["explanation"]
    assert "policy rate" not in s["explanation"]
    assert "above-median US payer" in s["message"]


@pytest.mark.parametrize("info", [
    {"pe_ratio": 1.0}, {"pe_ratio": 10.0}, {"pe_ratio": 60.0},
    {"dividend_yield": 3.0}, {"dividend_yield": 20.0},
    {"loss_making": True},
])
def test_no_us_signal_mentions_egx_or_egypt(info):
    for s in fundamental_signals("AAPL", info, US, 4.2):
        text = f"{s['message']} {s['explanation']}"
        assert "EGX" not in text and "Egypt" not in text and "EGP" not in text, text


@pytest.mark.parametrize("info", [{"pe_ratio": 1.0}, {"pe_ratio": 5.0}, {"pe_ratio": 60.0},
                                  {"dividend_yield": 3.0}, {"dividend_yield": 20.0}])
def test_signal_copy_is_condition_not_an_instruction(info):
    import re
    for m in (EG, US):
        for s in fundamental_signals("X", info, m, 4.2):
            text = f"{s['message']} {s['explanation']}".lower()
            assert not re.search(r"\b(sell|strong buy)\b", text), text


# ---------------------------------------------------------------------------
# Accuracy: the EG-measured causes are not asserted for US
# ---------------------------------------------------------------------------

# Real US names in the extreme bands on 2026-10-16 (pe_data, US): healthy large
# caps, and a REIT whose high yield is structural.
US_LOW_PE = [("ALL", 4.8), ("T", 7.77), ("HON", 7.78), ("CMCSA", 7.05)]


@pytest.mark.parametrize("symbol, pe", US_LOW_PE)
def test_eg_only_low_pe_concept_does_not_exist_on_us(symbol, pe):
    """No overlap: an EG-measured concept is absent on US, not adapted, and the
    stock does not fall through to 'cheap versus the US median'."""
    assert fundamental_signals(symbol, {"pe_ratio": pe}, US, 4.2) == []


def test_eg_only_extreme_yield_concept_does_not_exist_on_us():
    assert fundamental_signals("VICI", {"dividend_yield": 8.02}, US, 4.2) == []
    assert fundamental_signals("X", {"dividend_yield": 30.0}, US, 4.2) == []


def test_us_keeps_the_concepts_that_fit_it():
    types = lambda info: [s["type"] for s in fundamental_signals("X", info, US, 4.2)]
    assert types({"pe_ratio": US_FUND.pe_bands[0][0]}) == ["pe_undervalued"]
    assert types({"pe_ratio": US_FUND.pe_bands[3][0]}) == ["pe_overvalued"]
    assert types({"dividend_yield": US_FUND.dy_bands[1][0]}) == ["dividend_yield_solid"]


def test_only_eg_asserts_the_extreme_band_causes():
    assert EG_FUND.extreme_causes_measured is True
    assert US_FUND.extreme_causes_measured is False


def test_us_score_reasons_make_no_unmeasured_cause_claim():
    from app.core.composite import score_quality
    _, low = score_quality(None, None, None, pe_ratio=4.8, fundamentals=US_FUND)
    _, high = score_quality(None, None, None, dividend_yield=9.1, fundamentals=US_FUND)
    text = " ".join(low + high)
    for claim in ("implausibly low", "typically a special dividend", "collapsed", "EGX"):
        assert claim not in text, (claim, text)
    assert "P/E 4.8 — far below the US median of ~24" in low
    assert "Very high yield (DY 9.1%) — far above the US median of ~2.0%" in high
    assert "one-off" not in text and "recurs" not in text


def test_eg_score_reasons_keep_their_measured_wording():
    from app.core.composite import score_quality
    _, low = score_quality(None, None, None, pe_ratio=0.7, fundamentals=EG_FUND)
    _, high = score_quality(None, None, None, dividend_yield=42.55, fundamentals=EG_FUND)
    assert "P/E 0.7 — implausibly low; check the earnings are recurring" in low
    assert ("Extreme yield (DY 42.5%) — typically a special dividend or a "
            "collapsed share price, not steady income") in high


def test_us_extreme_bands_add_no_points_eg_keeps_its_shape():
    """The extreme bands' points encode EG-measured causes: 0 on US, unchanged on EG."""
    from app.core.composite import score_quality
    # a neutral drawdown (-10%) so the category has a score to compare
    base, _ = score_quality(None, None, -10.0, fundamentals=US_FUND)
    assert score_quality(None, None, -10.0, pe_ratio=4.8, fundamentals=US_FUND)[0] == base
    assert score_quality(None, None, -10.0, dividend_yield=9.1, fundamentals=US_FUND)[0] == base
    # Bands that fit both markets still score on US.
    assert score_quality(None, None, -10.0, pe_ratio=10.0, fundamentals=US_FUND)[0] == base + 12
    eg_base, _ = score_quality(None, None, -10.0, fundamentals=EG_FUND)
    assert score_quality(None, None, -10.0, pe_ratio=0.7, fundamentals=EG_FUND)[0] == eg_base + 3
    assert score_quality(None, None, -10.0, dividend_yield=42.55, fundamentals=EG_FUND)[0] == eg_base - 8
