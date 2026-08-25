"""
Vendored copy of `egxpy` v1.1.0 (commit 39703a3492f2a5f572c3c964c47d70e7bf4822ae).

WHY THIS IS VENDORED
--------------------
`requirements.txt` used to install this from
`git+https://github.com/egxlytics/egxpy.git`. That repository is no longer
reachable — GitHub returns "Repository not found" even to an authenticated
account that had previously installed from it, so the owner deleted it or
made it private. There is no PyPI release to fall back on.

Vercel builds failed at dependency resolution with:

    fatal: could not read Username for 'https://github.com': terminal prompts disabled

Deploys had been surviving on a warm build cache that still held an
already-cloned copy; the first cold build exposed it.

This is the last copy we have. It is reproduced verbatim (see download.py)
under its MIT licence — LICENSE in this directory, (c) 2025 EGXLytics.

MAINTENANCE
-----------
There is no upstream to pull from any more, so treat this as our code now.
It is a thin wrapper over `tvDatafeed` (which IS still public and installs
from git normally). If it needs changing, edit download.py directly and note
the deviation from v1.1.0 here.

Only three functions are used by the app:
  - get_OHLCV_data          (analysis, ohlcv, portfolio_analysis, macro_fetch)
  - get_EGXdata             (compare, historical)
  - get_EGX_intraday_data   (intraday)
"""

from app.vendor.egxpy.download import (  # noqa: F401
    get_EGX_intraday_data,
    get_EGXdata,
    get_OHLCV_data,
)

__all__ = ["get_OHLCV_data", "get_EGXdata", "get_EGX_intraday_data"]
__version__ = "1.1.0+vendored"
