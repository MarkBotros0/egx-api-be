"""
The realized ledger — what was closed, what was paid out, and what that came to.

POST   /api/sales          — record a full or partial sell
GET    /api/sales          — closed trades AND dividends, plus the combined summary
DELETE /api/sales?id=xxx   — undo a sale, restoring the shares

The GET serves BOTH ledgers on purpose. The Winnings headline is capital gains
plus dividends, and computing that sum here keeps it in tested Python — done in
the browser it would be the one number on the page with no test behind it.
Writes are split: sales here, dividends in routers/dividends.py.

Deliberately separate from /api/portfolio_analysis, which is the heaviest
endpoint in the app and flirts with the 30 s Vercel timeout. Neither ledger
needs a price fetch, so the Winnings card paints even on a run where the
analysis times out.

Every route is scoped by the caller's user_id from the JWT.
"""

import uuid
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import CurrentUser, get_current_user
from app.core.constants import DEFAULT_RISK_FREE_RATE_PCT
from app.core.db import get_db
from app.core.holdings import HOLDING_COLUMNS, fetch_open_lots, row_to_holding
from app.core.dividends import fetch_dividends, summarize_realized
from app.core.macro_series import get_risk_free_steps
from app.core.sales import (
    SaleValidationError,
    compute_sale_metrics,
    group_sale_orders,
    validate_position_sale,
)

router = APIRouter()

_SALE_COLUMNS = (
    "id, holding_id, symbol, name, sector, quantity, buy_price, buy_date, "
    "sell_price, sell_date, notes, created_at, sale_group_id"
)


def _row_to_sale(row) -> dict:
    return {
        "id": row[0], "holding_id": row[1], "symbol": row[2], "name": row[3],
        "sector": row[4], "quantity": row[5], "buy_price": row[6],
        "buy_date": row[7], "sell_price": row[8], "sell_date": row[9],
        "notes": row[10], "created_at": row[11], "sale_group_id": row[12],
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
    """
    Record a sell against a POSITION, which may span several purchase lots.

    `holding_id` identifies the position — its symbol — and `quantity` may
    exceed what that one row holds: someone who bought 200 then 100 owns 300
    shares and can sell any number up to it. The lots are consumed oldest
    first and each one consumed writes its OWN portfolio_sales row (see
    `sales.plan_sale_allocation` for why a blended row would be a lie).

    Every decrement and insert runs inside ONE transaction, so a sale spanning
    two lots cannot half-land.
    """
    try:
        holding_id = body.get("holding_id")
        if not holding_id:
            raise HTTPException(status_code=400, detail="Missing required field: holding_id")

        db = get_db()

        # Deliberately NOT fetch_open_holdings: this is a by-id lookup and it
        # must see the row regardless of remaining quantity so the error
        # message can say "you hold 0 shares" rather than "not found". It
        # resolves the SYMBOL; the shares come from the lot set below, so a
        # sale can still be recorded from a row that is itself fully closed.
        row = db.execute(
            f"SELECT {HOLDING_COLUMNS} FROM portfolio WHERE id = %s AND user_id = %s",
            (holding_id, user.id),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Holding not found: {holding_id}")
        anchor = row_to_holding(row)

        lots = fetch_open_lots(db, user.id, anchor["symbol"])

        clean = validate_position_sale(
            lots=lots,
            quantity=body.get("quantity"),
            sell_price=body.get("sell_price"),
            sell_date=body.get("sell_date"),
            today=date.today(),
        )

        now = datetime.utcnow().isoformat() + "Z"
        notes = body.get("notes", "")
        # One submit, one order id, however many lots it reaches into. The
        # ledger groups on this, so it is stamped even on a single-lot sale —
        # a row that has one is unambiguous, and a NULL only ever means
        # "written before orders existed".
        group_id = str(uuid.uuid4())
        sales: list[dict] = []
        remaining_by_lot: dict[str, int] = {}

        with db.transaction() as tx:
            for part in clean["allocation"]:
                # `quantity >= %s` in the WHERE clause makes the decrement
                # itself the over-sell guard, so two rapid submits cannot both
                # succeed — and it guards every lot, not just the first.
                updated = tx.execute(
                    "UPDATE portfolio SET quantity = quantity - %s, updated_at = %s "
                    "WHERE id = %s AND user_id = %s AND quantity >= %s "
                    "RETURNING quantity",
                    (part["quantity"], now, part["id"], user.id, part["quantity"]),
                ).fetchone()
                if updated is None:
                    # The plan was built from rows read above, so reaching here
                    # means a concurrent sell moved the count underneath us.
                    # Quoting the lot's own count would print "You hold 100
                    # shares — you cannot sell 100."
                    raise SaleValidationError(
                        "Not enough shares remaining — refresh and try again."
                    )

                sale_id = str(uuid.uuid4())
                tx.execute(
                    f"INSERT INTO portfolio_sales ({_SALE_COLUMNS}, user_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        sale_id, part["id"], part["symbol"], part["name"],
                        part.get("sector") or "", part["quantity"],
                        float(part["buy_price"]), part["buy_date"],
                        clean["sell_price"], clean["sell_date"],
                        notes, now, group_id, user.id,
                    ),
                )
                remaining_by_lot[part["id"]] = updated[0]
                sales.append({
                    "id": sale_id, "holding_id": part["id"], "symbol": part["symbol"],
                    "name": part["name"], "sector": part.get("sector") or "",
                    "quantity": part["quantity"], "buy_price": float(part["buy_price"]),
                    "buy_date": part["buy_date"], "sell_price": clean["sell_price"],
                    "sell_date": clean["sell_date"], "notes": notes,
                    "created_at": now, "sale_group_id": group_id,
                })

        # Read ONCE for the whole request, not once per lot consumed.
        rfr = _risk_free_rate_pct(db)
        rate_steps = get_risk_free_steps(db)
        priced = [compute_sale_metrics(s, rfr, rate_steps=rate_steps) for s in sales]
        return {
            "sales": priced,
            # What the user placed — one order, however many rows it wrote.
            "orders": group_sale_orders(priced),
            "holdings": [
                {**lot, "quantity": remaining_by_lot[lot["id"]], "updated_at": now}
                for lot in lots if lot["id"] in remaining_by_lot
            ],
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
        # Read ONCE, outside the comprehension — the steps are the same for
        # every trade, and a ledger of fifty closed positions must not make
        # fifty macro queries.
        rate_steps = get_risk_free_steps(db)
        priced = [compute_sale_metrics(_row_to_sale(r), rfr, rate_steps=rate_steps)
                  for r in rows]
        dividends = fetch_dividends(db, user.id)

        currency_row = db.execute(
            "SELECT value FROM settings WHERE key = 'currency'"
        ).fetchone()

        return {
            # The flat per-lot rows. `summarize_realized` eats THESE, not the
            # orders: it grades each trade against the rate that prevailed over
            # its own window, which is exactly what an order cannot state.
            "sales": priced,
            # The same rows folded back into what the user actually placed.
            "orders": group_sale_orders(priced),
            "dividends": dividends,
            "summary": summarize_realized(priced, dividends),
            "currency": currency_row[0] if currency_row else "EGP",
            "risk_free_rate_pct": rfr,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/sales")
def delete_sale(id: str = Query(...), user: CurrentUser = Depends(get_current_user)):
    """
    Undo a sale, restoring the shares to the lots they came from.

    `id` is an ORDER id or a single row's id, and the WHERE clause accepts
    either. That is deliberate rather than lax: the ledger shows one line per
    submit, so undoing what is on screen has to undo the whole thing — a delete
    that removed one part of a two-lot order would give back half the shares
    and leave the line still sitting there. The expanded row can still address
    one part by its own id when that is genuinely what is wanted.

    Every row's restore runs in ONE transaction, so a two-lot undo cannot
    half-land any more than the sale that wrote it could.
    """
    try:
        db = get_db()
        with db.transaction() as tx:
            rows = tx.execute(
                "DELETE FROM portfolio_sales "
                "WHERE (id = %s OR sale_group_id = %s) AND user_id = %s "
                "RETURNING id, holding_id, quantity",
                (id, id, user.id),
            ).fetchall()
            if not rows:
                raise HTTPException(status_code=404, detail=f"Sale not found: {id}")

            now = datetime.utcnow().isoformat() + "Z"
            restored = []
            for row_id, holding_id, quantity in rows:
                # If the user hard-deleted the holding, there is nothing to
                # restore the shares to — the sale still goes, and this part
                # reports None rather than failing the whole undo.
                back = tx.execute(
                    "UPDATE portfolio SET quantity = quantity + %s, updated_at = %s "
                    "WHERE id = %s AND user_id = %s RETURNING quantity",
                    (int(quantity), now, holding_id, user.id),
                ).fetchone()
                restored.append({
                    "sale_id": row_id,
                    "holding_id": holding_id,
                    "quantity": int(quantity),
                    "restored_quantity": back[0] if back else None,
                })

        return {"deleted": id, "restored": restored}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
