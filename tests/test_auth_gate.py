"""
The app-wide auth gate: nothing is served without a sign-in.

Run from egx-api-be:  python -m pytest tests/test_auth_gate.py -v

These tests walk the app's ACTUAL route table rather than a hand-written list,
so a router added later is covered the moment it is wired up. That is the whole
point of the gate: nine endpoints (tickers, ohlcv, analysis, compare,
historical, intraday, macro, market_regime, pe) served the full dataset to
anyone on the internet until it existed, and nothing in the codebase noticed.
"""

import os

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute

os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-gate")
os.environ.setdefault("DATABASE_URL", "postgresql://unused/unused")

from app.core import auth as auth_mod
from app.core.auth import (
    PUBLIC_ENDPOINTS,
    CurrentUser,
    authenticate_request,
    create_access_token,
    is_public_endpoint,
)


# ---------------------------------------------------------------------------
# The allowlist itself
# ---------------------------------------------------------------------------

def test_only_login_and_the_crons_are_public():
    """
    Adding an entry here means opening a hole in the app. It should take a
    deliberate edit and a failing test, never a quiet import.

    Every scheduled entry MUST carry its own shared-secret guard, because the
    middleware stops checking once a route is on this list:
      /api/pe/refresh        -> PE_REFRESH_SECRET
      /api/cron/risk_snapshot -> CRON_SECRET
    """
    assert PUBLIC_ENDPOINTS == frozenset({
        ("POST", "/api/auth/login"),
        ("POST", "/api/pe/refresh"),
        ("POST", "/api/cron/risk_snapshot"),
    })


def test_every_public_cron_checks_a_shared_secret():
    """
    The allowlist removes the token requirement; without a secret check in the
    handler the route would be open to the whole internet. This walks the
    source of each scheduled route and fails if one stops reading an env var.
    """
    import inspect

    from app.routers import cron as cron_mod
    from app.routers import pe as pe_mod

    for module, env_var in ((pe_mod, "PE_REFRESH_SECRET"),
                            (cron_mod, "CRON_SECRET")):
        src = inspect.getsource(module)
        assert f'os.environ.get("{env_var}")' in src, (
            f"{module.__name__} is on the public allowlist but no longer reads "
            f"{env_var} — that route is open to anyone"
        )


def test_login_is_reachable_without_a_token():
    assert is_public_endpoint("POST", "/api/auth/login")
    assert authenticate_request("POST", "/api/auth/login", None) is None


def test_the_cron_refresh_is_reachable_without_a_user_token():
    """Vercel's cron has no user; PE_REFRESH_SECRET is what guards this one."""
    assert is_public_endpoint("POST", "/api/pe/refresh")
    assert authenticate_request("POST", "/api/pe/refresh", None) is None


def test_preflights_pass_so_cors_still_works():
    """
    An OPTIONS preflight carries no Authorization header by design. Blocking it
    would break every browser call including the login request itself.
    """
    assert is_public_endpoint("OPTIONS", "/api/portfolio")
    assert is_public_endpoint("OPTIONS", "/api/anything/at/all")


def test_the_method_matters_not_just_the_path():
    """POST /api/pe/refresh is open; GET /api/pe is not."""
    assert is_public_endpoint("POST", "/api/pe/refresh")
    assert not is_public_endpoint("GET", "/api/pe/refresh")
    assert not is_public_endpoint("GET", "/api/pe")


def test_a_trailing_slash_does_not_open_a_closed_path():
    assert not is_public_endpoint("GET", "/api/portfolio/")


# ---------------------------------------------------------------------------
# Every route in the real app
# ---------------------------------------------------------------------------

def _app_routes():
    from app.main import app
    out = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if not route.path.startswith("/api/"):
            continue
        for method in route.methods:
            if method in ("HEAD", "OPTIONS"):
                continue
            out.append((method, route.path))
    return sorted(set(out))


def test_the_app_actually_has_routes():
    """Guards the tests below from passing vacuously on an import failure."""
    assert len(_app_routes()) > 15


def test_every_route_except_the_allowlist_rejects_an_anonymous_caller():
    leaks = []
    for method, path in _app_routes():
        if is_public_endpoint(method, path):
            continue
        try:
            authenticate_request(method, path, None)
        except HTTPException as e:
            if e.status_code == 401:
                continue
            leaks.append(f"{method} {path} -> {e.status_code}")
        else:
            leaks.append(f"{method} {path} -> served anonymously")

    assert not leaks, "these routes are open to the internet: " + ", ".join(leaks)


def test_the_endpoints_that_used_to_be_public_are_now_closed():
    """
    Named explicitly, because this is the regression that matters: each of
    these served real market data to anyone who asked.
    """
    for path in (
        "/api/tickers", "/api/ohlcv", "/api/analysis", "/api/compare",
        "/api/historical", "/api/intraday", "/api/macro",
        "/api/market_regime", "/api/pe",
    ):
        with pytest.raises(HTTPException) as e:
            authenticate_request("GET", path, None)
        assert e.value.status_code == 401, path


def test_a_garbage_token_is_rejected():
    with pytest.raises(HTTPException) as e:
        authenticate_request("GET", "/api/tickers", "Bearer not-a-real-token")
    assert e.value.status_code == 401


def test_a_missing_bearer_prefix_is_rejected():
    with pytest.raises(HTTPException) as e:
        authenticate_request("GET", "/api/tickers", "some-token-without-bearer")
    assert e.value.status_code == 401


def test_a_valid_token_for_an_active_user_passes(monkeypatch):
    monkeypatch.setattr(
        auth_mod, "_load_user",
        lambda uid: CurrentUser(id=uid, username="sara", role="user"),
    )
    token = create_access_token("user-1", "sara")
    user = authenticate_request("GET", "/api/tickers", f"Bearer {token}")
    assert user.id == "user-1"


def test_a_valid_token_for_a_DISABLED_user_is_still_rejected(monkeypatch):
    """The gate must not be weaker than the per-route dependency it backs up."""
    monkeypatch.setattr(auth_mod, "_load_user", lambda uid: None)
    token = create_access_token("user-1", "sara")
    with pytest.raises(HTTPException) as e:
        authenticate_request("GET", "/api/tickers", f"Bearer {token}")
    assert e.value.status_code == 401


# ---------------------------------------------------------------------------
# The middleware is actually wired up
# ---------------------------------------------------------------------------

def test_main_installs_the_gate_middleware():
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "app" / "main.py"
    ).read_text(encoding="utf-8")
    assert "authenticate_request" in src, "main.py must run the gate"
    assert '@app.middleware("http")' in src


def test_analysis_no_longer_accepts_an_anonymous_caller():
    """
    /api/analysis used get_optional_user because the dashboard was a public
    page. It is not public any more, and leaving the optional dependency would
    mean the route silently scored under DEFAULT weights if the gate were ever
    relaxed.
    """
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "app" / "routers" / "analysis.py"
    ).read_text(encoding="utf-8")
    assert "get_optional_user" not in src
    assert "Depends(get_current_user)" in src
