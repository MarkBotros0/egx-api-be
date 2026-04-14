"""
NaN-safe JSON response for FastAPI.

Replaces the old _response.NaNSafeEncoder + send_json pattern.
Register NaNSafeJSONResponse as the default_response_class on the FastAPI app
so all route return values go through this encoder automatically.
"""

import json
import math
from starlette.responses import JSONResponse


def _sanitize(obj):
    """Recursively replace float NaN/Inf with None for JSON safety."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


class NaNSafeJSONResponse(JSONResponse):
    """JSONResponse subclass that converts NaN/Inf → null before serialization."""

    def render(self, content) -> bytes:
        return json.dumps(
            _sanitize(content),
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")
