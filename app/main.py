"""
EGX Analytics FastAPI application factory.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.auth import authenticate_request, seed_users_from_env
from app.core.db import get_db
from app.core.json_encoding import NaNSafeJSONResponse
from app.routers import (
    auth,
    users,
    tickers,
    ohlcv,
    macro,
    settings,
    watchlist,
    portfolio,
    compare,
    historical,
    intraday,
    analysis,
    portfolio_analysis,
    sales,
    dividends,
    pe,
    market_regime,
)

app = FastAPI(
    title="EGX Analytics API",
    description="Backend for the Egyptian Exchange stock analysis app.",
    version="1.0.0",
    default_response_class=NaNSafeJSONResponse,
)

_raw_origins = os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]


@app.middleware("http")
async def require_authentication(request: Request, call_next):
    """
    The whole API is closed. Nothing is served without a valid, active user
    except the routes in auth.PUBLIC_ENDPOINTS.

    A single chokepoint rather than a dependency on each router: this way a
    router added later is locked by DEFAULT, and forgetting to open it fails
    loudly instead of leaking silently. Routers that need the caller's identity
    still declare Depends(get_current_user) — this is a second layer, not a
    replacement, and it costs one extra row read on paths that do both.
    """
    if request.url.path.startswith("/api/"):
        try:
            authenticate_request(
                request.method,
                request.url.path,
                request.headers.get("authorization"),
            )
        except HTTPException as e:
            # Middleware sits outside FastAPI's exception handling, so the
            # response has to be built here rather than raised.
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

    return await call_next(request)


# Added AFTER the auth middleware, so it runs FIRST (Starlette wraps in
# reverse). Preflights must get their CORS headers even though they carry no
# Authorization header, and a 401 must still be readable by the browser
# instead of surfacing as an opaque CORS error.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

try:
    seed_users_from_env(get_db())
except Exception as e:
    # Never let a seeding failure crash the app — log and continue.
    print(f"[auth] seed_users_from_env failed: {e}")

for router_module in (
    auth,
    users,
    tickers,
    ohlcv,
    macro,
    settings,
    watchlist,
    portfolio,
    compare,
    historical,
    intraday,
    analysis,
    portfolio_analysis,
    sales,
    dividends,
    pe,
    market_regime,
):
    app.include_router(router_module.router)
