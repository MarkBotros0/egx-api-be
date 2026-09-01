"""
The one place the "open holdings" query is spelled.

A holding with quantity 0 is fully sold. The row is RETAINED so a sale can be
undone against it — restoring the position with its target price, stop loss
and notes intact — but it must never appear in the portfolio list or reach
portfolio analysis, where it would show as a phantom position holding zero
shares.

portfolio.py and portfolio_analysis.py both read the table directly
(portfolio_analysis does not call /api/portfolio), so the filter lives here
and both call it. tests/test_sell_tracking.py fails if either grows its own.

By-id lookups stay inline in their routers: PUT and DELETE need them, and
sales.py needs one that deliberately IGNORES this filter so a sale can be
undone against a fully-closed holding.
"""

HOLDING_COLUMNS = (
    "id, symbol, name, buy_price, buy_date, quantity, notes, sector, "
    "target_price, stop_loss, created_at, updated_at"
)


def row_to_holding(row) -> dict:
    """Map a HOLDING_COLUMNS row tuple to the holding dict the API returns."""
    return {
        "id": row[0],
        "symbol": row[1],
        "name": row[2],
        "buy_price": row[3],
        "buy_date": row[4],
        "quantity": row[5],
        "notes": row[6],
        "sector": row[7],
        "target_price": row[8],
        "stop_loss": row[9],
        "created_at": row[10],
        "updated_at": row[11],
    }


def fetch_open_holdings(db, user_id: str) -> list[dict]:
    """Every holding the user still owns at least one share of."""
    rows = db.execute(
        f"SELECT {HOLDING_COLUMNS} FROM portfolio "
        "WHERE user_id = %s AND quantity > 0 ORDER BY created_at",
        (user_id,),
    ).fetchall()
    return [row_to_holding(r) for r in rows]
