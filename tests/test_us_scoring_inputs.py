"""
Per-market scoring inputs: risk-free rate, macro regime and index tier.

A US stock must never be scored against Egypt's policy rate, modulated by the
EGX30 regime, or held to an EGX index's liquidity floor. EG must be unchanged.
"""

from alert_fakes import FakeDB
from app.core.markets import EG, US
from app.core import macro_fetch as M
from app.core.index_membership import get_index_membership
from app.core.indicators import liquidity_score
import pandas as pd


def test_us_risk_free_from_settings_or_fallback():
    db = FakeDB()
    db.rows_for("risk_free_rate:US", [("4.216",)])
    assert M.read_risk_free_rate(db, US) == 4.216
    assert M.read_risk_free_rate(FakeDB(), US) == 4.0
    assert M.read_risk_free_rate(None, US) == 4.0


def test_eg_risk_free_unchanged():
    db = FakeDB()
    db.rows_for("WHERE key = 'risk_free_rate'", [("21.5",)])
    assert M.read_risk_free_rate(db) == 21.5


def test_us_risk_free_never_reads_the_eg_key():
    db = FakeDB()
    db.rows_for("WHERE key = 'risk_free_rate'", [("21.5",)])
    assert M.read_risk_free_rate(db, US) == 4.0


def test_macro_only_for_eg(monkeypatch):
    monkeypatch.setattr(M, "fetch_macro", lambda db: {"egx30": {"trend": "up"}})
    assert M.fetch_macro_for(FakeDB(), US) is None
    assert M.fetch_macro_for(FakeDB(), EG) == {"egx30": {"trend": "up"}}


def test_membership_never_borrows_eg_tiers():
    assert get_index_membership("COMI", EG) == "EGX30"
    assert get_index_membership("COMI", US) == "US"


def test_us_liquidity_uses_lenient_floor():
    vol = pd.Series([10_000.0] * 30)
    out = liquidity_score(vol, index_membership="US")
    assert out["index_membership"] == "US" and out["classification"] == "low"


def test_us_tier_copy_grammar():
    from app.core.composite import _tier_article  # helper added in this task
    assert _tier_article("US") == "a" and _tier_article("EGX30") == "an" and _tier_article("NILEX") == "a"
    assert _tier_article("EGX70") == "an" and _tier_article("EGX100") == "an"


def _context(monkeypatch, market):
    from app.routers import analysis as analysis_mod
    db = FakeDB()
    db.rows_for("risk_free_rate:US", [("4.216",)])
    db.rows_for("WHERE key = 'risk_free_rate'", [("21.5",)])
    monkeypatch.setattr(analysis_mod, "get_db", lambda: db)
    monkeypatch.setattr(M, "fetch_macro", lambda db: {"egx30": {"trend": "down"}})
    return analysis_mod.scoring_cache_context(None, market)


def test_us_scoring_cache_context_uses_us_rate_and_no_macro(monkeypatch):
    _, _, macro, rfr, tags = _context(monkeypatch, US)
    assert macro is None
    assert rfr == 4.216
    assert "rfr4.216" in tags
    assert tags[1] == "n/a"


def test_eg_scoring_cache_context_unchanged(monkeypatch):
    _, _, macro, rfr, tags = _context(monkeypatch, EG)
    assert macro == {"egx30": {"trend": "down"}}
    assert rfr == 21.5
    assert tags[1] == "down" and tags[2] == "rfr21.5"
