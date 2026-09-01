"""
Sales ledger — record what was sold and report realized gains.

POST   /api/sales          — record a full or partial sell
GET    /api/sales          — every sale plus the realized-gains summary
DELETE /api/sales?id=xxx   — undo a sale, restoring the shares

Deliberately separate from /api/portfolio_analysis, which is the heaviest
endpoint in the app and flirts with the 30 s Vercel timeout. Realized gains
need NO price fetch, so the Winnings card paints even on a run where the
analysis times out.

Every route is scoped by the caller's user_id from the JWT.
"""

import uuid
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import CurrentUser, get_current_user
from app.core.constants import DEFAULT_RISK_FREE_RATE_PCT
from app.core.db import get_db
from app.core.holdings import HOLDING_COLUMNS, row_to_holding
from app.core.sales import (
    SaleValidationError,
    compute_sale_metrics,
    summarize_sales,
    validate_sale,
)

router = APIRouter()

_SALE_COLUMNS = (
    "id, holding_id, symbol, name, sector, quantity, buy_price, buy_date, "
    "sell_price, sell_date, notes, created_at"
)


def _row_to_sale(row) -> dict:
    return {
        "id": row[0], "holding_id": row[1], "symbol": row[2], "name": row[3],
        "sector": row[4], "quantity": row[5], "buy_price": row[6],
        "buy_date": row[7], "sell_price": row[8], "sell_date": row[9],
        "notes": row[10], "created_at": row[11],
    }


def _risk_free_rate_pct(db) -> float:
    try:
        row = db.execute(
            "SELECT value FROM settings WHERE key = 'risk_free_rate'"
        ).fetchone()
        return float(row[0]) if row else DEFAULT_RISK_FREE_RATE_PCT
    except Exception:
        return DEFAULT_RISK_FREE_RATE_PCT


@router.post("/api/sales", status_code=201)
def record_sale(body: dict, user: CurrentUser = Depends(get_current_user)):
    try:
        holding_id = body.get("holding_id")
        if not holding_id:
            raise HTTPException(status_code=400, detail="Missing required field: holding_id")

        db = get_db()

        # Deliberately NOT fetch_open_holdings: this is a by-id lookup and it
        # must see the row regardless of remaining quantity so the error
        # message can say "you hold 0 shares" rather than "not found".
        row = db.execute(
            f"SELECT {HOLDING_COLUMNS} FROM portfolio WHERE id = %s AND user_id = %s",
            (holding_id, user.id),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Holding not found: {holding_id}")
        holding = row_to_holding(row)

        clean = validate_sale(
            holding=holding,
            quantity=body.get("quantity"),
            sell_price=body.get("sell_price"),
            sell_date=body.get("sell_date"),
            today=date.today(),
        )

        now = datetime.utcnow().isoformat() + "Z"
        sale_id = str(uuid.uuid4())

        with db.transaction() as tx:
            # `quantity >= %s` in the WHERE clause makes the decrement itself
            # the over-sell guard, so two rapid submits cannot both succeed.
            updated = tx.execute(
                "UPDATE portfolio SET quantity = quantity - %s, updated_at = %s "
                "WHERE id = %s AND user_id = %s AND quantity >= %s "
                "RETURNING quantity",
                (clean["quantity"], now, holding_id, user.id, clean["quantity"]),
            ).fetchone()
            if updated is None:
                # `validate_sale` already checked the quantity against the row
                # read above, so reaching here means a concurrent sell moved
                # the count underneath us. Quoting `holding['quantity']` would
                # print "You hold 100 shares — you cannot sell 100."
                raise SaleValidationError(
                    "Not enough shares remaining — refresh and try again."
                )

            tx.execute(
                f"INSERT INTO portfolio_sales ({_SALE_COLUMNS}, user_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    sale_id, holding_id, holding["symbol"], holding["name"],
                    holding.get("sector") or "", clean["quantity"],
                    float(holding["buy_price"]), holding["buy_date"],
                    clean["sell_price"], clean["sell_date"],
                    body.get("notes", ""), now, user.id,
                ),
            )
            remaining = updated[0]

        sale = {
            "id": sale_id, "holding_id": holding_id, "symbol": holding["symbol"],
            "name": holding["name"], "sector": holding.get("sector") or "",
            "quantity": clean["quantity"], "buy_price": float(holding["buy_price"]),
            "buy_date": holding["buy_date"], "sell_price": clean["sell_price"],
            "sell_date": clean["sell_date"], "notes": body.get("notes", ""),
            "created_at": now,
        }
        return {
            "sale": compute_sale_metrics(sale, _risk_free_rate_pct(db)),
            "holding": {**holding, "quantity": remaining, "updated_at": now},
        }

    except SaleValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/sales")
def get_sales(user: CurrentUser = Depends(get_current_user)):
    try:
        db = get_db()
        rows = db.execute(
            f"SELECT {_SALE_COLUMNS} FROM portfolio_sales "
            "WHERE user_id = %s ORDER BY sell_date DESC, created_at DESC",
            (user.id,),
        ).fetchall()

        rfr = _risk_free_rate_pct(db)
        priced = [compute_sale_metrics(_row_to_sale(r), rfr) for r in rows]

        currency_row = db.execute(
            "SELECT value FROM settings WHERE key = 'currency'"
        ).fetchone()

        return {
            "sales": priced,
            "summary": summarize_sales(priced),
            "currency": currency_row[0] if currency_row else "EGP",
            "risk_free_rate_pct": rfr,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/sales")
def delete_sale(id: str = Query(...), user: CurrentUser = Depends(get_current_user)):
    try:
        db = get_db()
        with db.transaction() as tx:
            row = tx.execute(
                "DELETE FROM portfolio_sales WHERE id = %s AND user_id = %s "
                "RETURNING holding_id, quantity",
                (id, user.id),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Sale not found: {id}")
            holding_id, quantity = row[0], int(row[1])

            # If the user hard-deleted the holding, there is nothing to restore
            # the shares to — the sale still goes, and restored stays None.
            restored = tx.execute(
                "UPDATE portfolio SET quantity = quantity + %s, updated_at = %s "
                "WHERE id = %s AND user_id = %s RETURNING quantity",
                (quantity, datetime.utcnow().isoformat() + "Z", holding_id, user.id),
            ).fetchone()

        return {
            "deleted": id,
            "holding_id": holding_id,
            "restored_quantity": restored[0] if restored else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
