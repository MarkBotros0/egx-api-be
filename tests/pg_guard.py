"""
Shared skip guard for the tests that run against a REAL Postgres.

Those tests DROP tables in the current schema of TEST_DATABASE_URL, so they only
run when that URL points at this machine (localhost / 127.0.0.1 / ::1). Set
PG_TESTS_ALLOW_REMOTE=1 to run them against another host on purpose.
"""

import os
from urllib.parse import urlparse

import pytest

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def skip_reason(url: str | None, allow_remote: str | None = None) -> str | None:
    """Why the PG tests must be skipped for this URL, or None if they may run."""
    if not url:
        return "TEST_DATABASE_URL not set"
    if allow_remote == "1":
        return None
    try:
        host = urlparse(url).hostname
    except ValueError:
        host = None
    if host not in LOCAL_HOSTS:
        return (f"TEST_DATABASE_URL host {host!r} is not local; these tests drop "
                "tables (set PG_TESTS_ALLOW_REMOTE=1 to override)")
    return None


_REASON = skip_reason(os.environ.get("TEST_DATABASE_URL"),
                      os.environ.get("PG_TESTS_ALLOW_REMOTE"))

# Use as the module's `pytestmark`.
requires_local_pg = pytest.mark.skipif(_REASON is not None, reason=_REASON or "")
