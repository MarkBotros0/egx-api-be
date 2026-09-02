"""
The USD lens: the same return in the currency you actually spend.

The one thing that must not regress here is the FORM of the conversion.
Subtracting a devaluation percentage from an EGP return is the intuitive
shortcut and it is wrong whenever either move is large — which, on this
currency, is always.
"""

from __future__ import annotations

from app.core.currency import (
    EGX30_TWENTY_YEAR,
    currency_drag_pct,
    dual_return,
    to_usd,
    usd_return_pct,
)


def test_the_conversion_is_not_a_subtraction():
    """
    A stock that doubled in EGP while the pound halved returned +100% in pounds
    and +20% in dollars. The naive "subtract the 40% devaluation" gives +60% —
    wrong by 40 points, and wrong in the flattering direction.
    """
    usd = usd_return_pct(100.0, 200.0, 30.0, 50.0)
    assert round(usd, 2) == 20.0

    devaluation = (50.0 - 30.0) / 50.0 * 100      # 40%
    naive = 100.0 - devaluation                    # 60%
    assert abs(naive - usd) > 30, (
        "the shortcut and the correct form now agree, which means the test "
        "case no longer exercises the difference"
    )


def test_a_stable_currency_leaves_the_return_alone():
    assert usd_return_pct(100.0, 150.0, 50.0, 50.0) == 50.0


def test_a_gain_in_egp_can_be_a_loss_in_usd():
    """The case the whole module exists for."""
    result = dual_return(100.0, 130.0, 30.0, 50.0)
    assert result["egp_pct"] == 30.0
    assert result["usd_pct"] < 0
    assert result["currency_drag_pct"] > 30


def test_currency_drag_is_reported_separately():
    """
    It is the figure a user is least likely to derive themselves and most
    likely to be surprised by, so it is never folded into either return.
    """
    assert currency_drag_pct(100.0, 20.0) == 80.0
    assert currency_drag_pct(None, 20.0) is None


def test_missing_fx_history_degrades_to_the_egp_figure_alone():
    """
    A holding bought before the FX backfill reaches must still show its EGP
    return rather than nothing — the caller should never have to branch.
    """
    result = dual_return(100.0, 130.0, None, None)
    assert result["egp_pct"] == 30.0
    assert result["usd_pct"] is None
    assert result["currency_drag_pct"] is None


def test_degenerate_inputs_return_none_rather_than_raising():
    assert usd_return_pct(0.0, 100.0, 30.0, 50.0) is None
    assert usd_return_pct(100.0, 200.0, 0.0, 50.0) is None
    assert usd_return_pct(None, 200.0, 30.0, 50.0) is None
    assert to_usd(100.0, 0.0) is None
    assert to_usd(None, 50.0) is None


def test_to_usd_divides_rather_than_multiplies():
    """USD/EGP is pounds per dollar, so converting DIVIDES. Inverting it here
    would overstate every figure by the square of the rate."""
    assert to_usd(510.0, 51.0) == 10.0


def test_the_twenty_year_context_is_the_measured_pair():
    """
    8.25x in EGP against 0.94x in USD is the fact the lens exists to surface.
    If these ever drift apart from the panel, the yardstick is lying.
    """
    assert EGX30_TWENTY_YEAR["egp_multiple"] > 8
    assert EGX30_TWENTY_YEAR["usd_multiple"] < 1
