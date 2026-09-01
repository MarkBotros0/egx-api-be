"""
Settings endpoint — Manage app settings (currency, risk_free_rate, composite weights, etc.)

GET  /api/settings                   — All settings as key/value map
PUT  /api/settings                   — Update a single setting {key, value}
GET  /api/settings?section=weights   — Composite-score weights
PUT  /api/settings?section=weights   — Update weights (normalized to 100)
"""

from typing import Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.auth import CurrentUser, get_current_user, require_admin
from app.core.db import get_db
from app.core.composite import (
    CATEGORY_ORDER,
    DEFAULT_WEIGHTS,
    PRESETS,
    get_weights_from_db,
    normalize_weights,
)

router = APIRouter()


class SettingUpdate(BaseModel):
    key: str
    value: Any


class WeightsUpdate(BaseModel):
    weights: dict


@router.get("/api/settings")
def get_settings(
    section: Optional[str] = Query(None),
    user: CurrentUser = Depends(get_current_user),
):
    try:
        if section == "weights":
            db = get_db()
            raw = get_weights_from_db(db, user.id)
            normalized = normalize_weights(raw)
            return {
                "weights": normalized,
                "raw": raw,
                "presets": PRESETS,
                "default": DEFAULT_WEIGHTS,
            }

        db = get_db()
        rows = db.execute("SELECT key, value FROM settings").fetchall()
        return {r[0]: r[1] for r in rows}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/settings")
def put_settings(
    body: dict,
    section: Optional[str] = Query(None),
    user: CurrentUser = Depends(get_current_user),
):
    try:
        if section == "weights":
            weights_in = body.get("weights")
            if not isinstance(weights_in, dict):
                raise HTTPException(status_code=400, detail="Missing required field: weights (object)")

            sanitized = {}
            for key in CATEGORY_ORDER:
                val = weights_in.get(key)
                if val is None:
                    continue
                try:
                    sanitized[key] = float(val)
                except (TypeError, ValueError):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid value for weight '{key}': not a number",
                    )

            if not sanitized:
                raise HTTPException(status_code=400, detail="No valid weight values provided")

            db = get_db()
            current = get_weights_from_db(db, user.id)
            current.update(sanitized)
            normalized = normalize_weights(current)

            # Written to user_settings, not settings: weights are a preference,
            # and before this each save silently rescored every other user's
            # stocks too.
            for key, value in normalized.items():
                db.execute(
                    "INSERT INTO user_settings (user_id, key, value) VALUES (%s, %s, %s) "
                    "ON CONFLICT (user_id, key) DO UPDATE SET value = EXCLUDED.value",
                    (user.id, f"weight_{key}", str(value)),
                )
            db.commit()
            return {"weights": normalized, "raw": current}

        # Everything else in `settings` is global and shared — currency, the
        # risk-free rate (also the CBE rate on the macro card and the bar
        # realized trades are graded against), and the P/E feed status. One
        # user must not be able to move those for everyone.
        _ = require_admin(user)

        key = body.get("key")
        value = body.get("value")

        if not key or value is None:
            raise HTTPException(status_code=400, detail="Missing required fields: key and value")

        db = get_db()
        db.execute(
            "INSERT INTO settings (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (key, str(value)),
        )
        db.commit()
        return {"key": key, "value": str(value)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
