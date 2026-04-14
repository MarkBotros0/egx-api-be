"""
EGX Analytics FastAPI application factory.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.json_encoding import NaNSafeJSONResponse
from app.routers import (
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

for router_module in (
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
