"""
Dividends — record cash a company paid you for holding it.

POST   /api/dividends          — record one
DELETE /api/dividends?id=xxx   — remove one

Reads are NOT here: dividends come back on GET /api/sales alongside closed
trades, so the Winnings card gets one fetch and the combined headline is
computed in tested Python rather than in the browser.

Unlike a sale, recording a dividend is a SINGLE statement — it decrements no
share count — so there is nothing for db.transaction() to keep atomic. Deleting
one restores nothing for the same reason.

Every route is scoped by the caller's user_id from the JWT. POST records into
a market (body `market` beats `?market=`, absent means EG); DELETE acts on a
row by id, which already carries its market.
"""

import uuid
from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import CurrentUser, get_current_user
from app.core.db import get_db
from app.core.markets import MarketConfig, get_market, market_param
from app.core.dividends import (
    DividendValidationError,
    enrich_dividend,
    fetch_dividends,
    is_duplicate,
    validate_dividend,
)

router = APIRouter()


@router.post("/api/dividends", status_code=201)
def record_dividend(body: dict,
                    market: MarketConfig = Depends(market_param),
                    user: CurrentUser = Depends(get_current_user)):
    try:
        target = get_market(body["market"]) if body.get("market") else market
        db = get_db()

        clean = validate_dividend(
            symbol=body.get("symbol"),
            amount=body.get("amount"),
            pay_date=body.get("pay_date"),
            shares=body.get("shares"),
            today=date.today(),
            share_decimals=target.share_decimals,
        )

        # A double-tapped submit on a phone is the likeliest way this ledger
        # goes silently wrong: a duplicate dividend corrupts no share count, so
        # nothing else would ever reveal it.
        # Scoped to the market: the same symbol + date + amount in another
        # market is a different payment.
        if is_duplicate(fetch_dividends(db, user.id, target.code), clean):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"You already recorded a dividend of {clean['amount']:.2f} {target.currency} "
                    f"for {clean['symbol']} on {clean['pay_date']}."
                ),
            )

        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"
        dividend_id = str(uuid.uuid4())
        name = body.get("name") or clean["symbol"]
        sector = body.get("sector") or ""
        notes = body.get("notes", "")

        db.execute(
            "INSERT INTO portfolio_dividends "
            "(id, user_id, market, symbol, name, sector, amount, pay_date, shares, notes, "
            "created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                dividend_id, user.id, target.code, clean["symbol"], name, sector,
                clean["amount"], clean["pay_date"], clean["shares"], notes, now,
            ),
        )

        return {
            "market": target.code,
            "currency": target.currency,
            "dividend": enrich_dividend({
                "id": dividend_id,
                "symbol": clean["symbol"],
                "name": name,
                "sector": sector,
                "amount": clean["amount"],
                "pay_date": clean["pay_date"],
                "shares": clean["shares"],
                "notes": notes,
                "created_at": now,
            })
        }

    except DividendValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/dividends")
def delete_dividend(id: str = Query(...), user: CurrentUser = Depends(get_current_user)):
    try:
        db = get_db()
        # RETURNING makes the delete its own existence check, so a 404 cannot
        # race a concurrent delete into a false success.
        row = db.execute(
            "DELETE FROM portfolio_dividends WHERE id = %s AND user_id = %s RETURNING id",
            (id, user.id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Dividend not found: {id}")
        return {"deleted": id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
