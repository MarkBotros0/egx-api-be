"""
The app's route table, read through FastAPI's PUBLIC surface.

The route tests used to walk `app.routes` for `APIRoute` objects. FastAPI
0.139+ wraps every `include_router` in a private `_IncludedRouter`, so that
walk found ZERO routes — and the auth-gate sweep built on it passed while
checking nothing (only its "> 15 routes" guard noticed). The OpenAPI schema
is the documented, version-stable listing of the same routes.

The one thing it cannot see is a route declared `include_in_schema=False`;
`test_auth_gate.py::test_no_route_hides_from_the_schema` keeps that from
ever existing, so this listing stays the whole table.
"""

from __future__ import annotations

_HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


def api_routes() -> list[tuple[str, str]]:
    """Sorted (METHOD, path) for every /api/ route the app serves."""
    from app.main import app

    out = set()
    for path, operations in app.openapi().get("paths", {}).items():
        if not path.startswith("/api/"):
            continue
        for method in operations:
            if method in _HTTP_METHODS:
                out.add((method.upper(), path))
    return sorted(out)


def api_paths() -> set[str]:
    return {path for _, path in api_routes()}
