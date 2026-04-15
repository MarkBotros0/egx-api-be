"""
EGX Analytics FastAPI application factory.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.auth import seed_users_from_env
from app.core.db import get_db
from app.core.json_encoding import NaNSafeJSONResponse
from app.routers import (
    auth,
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
)

app = FastAPI(
    title="EGX Analytics API",
    description="Backend for the Egyptian Exchange stock analysis app.",
    version="1.0.0",
    default_response_class=NaNSafeJSONResponse,
)

_raw_origins = os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

try:
    seed_users_from_env(get_db())
except Exception as e:
    # Never let a seeding failure crash the app — log and continue.
    print(f"[auth] seed_users_from_env failed: {e}")

for router_module in (
    auth,
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
):
    app.include_router(router_module.router)
