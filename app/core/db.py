"""
Neon Postgres database connection helper.

Exposes get_db() returning a thin wrapper over a psycopg connection pool.
The wrapper preserves the historical `db.execute(sql, params).fetchone()/fetchall()`
and `db.commit()` surface so routers did not need structural changes when moving
off Turso. Connections run with autocommit=True, so `commit()` is a no-op.
"""

import os
from contextlib import contextmanager
from psycopg_pool import ConnectionPool

from app.core.constants import (
    DEFAULT_RISK_FREE_RATE_PCT,
    STALE_RISK_FREE_RATE_PCT,
)

_pool = None
_initialized = False


class _Result:
    __slots__ = ("_rows",)

    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class _Tx:
    """Statement executor bound to one connection inside a transaction."""

    __slots__ = ("_conn",)

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql: str, params=()):
        cur = self._conn.execute(sql, params)
        rows = cur.fetchall() if cur.description else []
        return _Result(rows)


class _DB:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def execute(self, sql: str, params=()):
        with self._pool.connection() as conn:
            cur = conn.execute(sql, params)
            rows = cur.fetchall() if cur.description else []
            return _Result(rows)

    def commit(self):
        # autocommit is enabled on pool connections; retained for API parity
        pass

    @contextmanager
    def transaction(self):
        """
        Run several statements on ONE connection inside a real transaction.

        execute() takes a fresh pooled connection per call and the pool runs
        with autocommit, so two related writes issued through it can half-land
        — recording a sale without decrementing the holding would invent
        shares. Anything that must be all-or-nothing goes through here.
        """
        with self._pool.connection() as conn:
            with conn.transaction():
                yield _Tx(conn)


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        url = os.environ.get("DATABASE_URL", "")
        if not url:
            raise RuntimeError("DATABASE_URL is not set")
        _pool = ConnectionPool(
            conninfo=url,
            min_size=1,
            max_size=5,
            kwargs={"autocommit": True},
            open=True,
        )
    return _pool


def init_db(db: _DB) -> None:
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    # Additive and idempotent, matching the rest of init_db — there is no
    # migration framework, so new columns land on the next cold start.
    #
    # `role` is NOT settable through the admin API: it is stamped from the
    # AUTH_ADMINS env var at boot (see core/auth.seed_users_from_env), which
    # makes privilege escalation through /api/users structurally impossible
    # and keeps admin status declared in one auditable place that survives a
    # DB reset.
    db.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'user'")
    db.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE")

    db.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            buy_price DOUBLE PRECISION NOT NULL,
            buy_date TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            notes TEXT DEFAULT '',
            sector TEXT DEFAULT '',
            target_price DOUBLE PRECISION,
            stop_loss DOUBLE PRECISION,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    db.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_user ON portfolio(user_id)")

    # Append-only ledger of sales. Cost basis is SNAPSHOTTED (buy_price,
    # buy_date, name, sector) rather than joined from portfolio: a sale is a
    # historical fact and must not change when the user later edits or deletes
    # the holding it came from. Same principle as fundamentals_history.
    db.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_sales (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            holding_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            sector TEXT DEFAULT '',
            quantity INTEGER NOT NULL,
            buy_price DOUBLE PRECISION NOT NULL,
            buy_date TEXT NOT NULL,
            sell_price DOUBLE PRECISION NOT NULL,
            sell_date TEXT NOT NULL,
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_portfolio_sales_user "
        "ON portfolio_sales(user_id)"
    )

    # Dividends — cash the company paid you for holding it.
    #
    # Symbol-anchored, deliberately with NO holding_id. A sale carries one
    # because DELETE /api/sales must restore shares to a specific position; a
    # dividend restores nothing, so the column would buy no behaviour — and
    # would cost correctness, since deleting a holding would then orphan or
    # destroy the record of money genuinely received.
    #
    # `amount` is the total EGP that ACTUALLY LANDED, already net of Egypt's
    # 5-10% dividend withholding tax. The app computes no tax and must never
    # present this as a gross figure. `shares` is optional and exists only so
    # the UI can show an approximate per-share number; amount is never derived
    # from it.
    db.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_dividends (
            id         TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            symbol     TEXT NOT NULL,
            name       TEXT NOT NULL,
            sector     TEXT DEFAULT '',
            amount     DOUBLE PRECISION NOT NULL,
            pay_date   TEXT NOT NULL,
            shares     INTEGER,
            notes      TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_portfolio_dividends_user "
        "ON portfolio_dividends(user_id, symbol)"
    )

    db.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    # Per-user overrides of the keys in `settings` that are genuinely a
    # PREFERENCE rather than a fact — today that is the composite weight_*
    # keys and nothing else.
    #
    # A separate table rather than a user_id column on `settings`: changing
    # that table's primary key is not idempotent in Postgres, and keeping
    # `settings` as the global tier gives the migration for free. The read
    # chain is user_settings -> settings -> DEFAULT_WEIGHTS per key
    # (composite.get_weights_from_db), so an existing install's saved weights
    # stay everyone's starting point and nobody's scores jump on deploy.
    #
    # risk_free_rate deliberately does NOT live here. It is the Sharpe hurdle,
    # the CBE policy rate the macro card renders, AND the bar realized trades
    # are graded against — a market fact, not a preference. Per-user values
    # would mean each user grading their own trades against a different bar.
    db.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id TEXT NOT NULL,
            key     TEXT NOT NULL,
            value   TEXT NOT NULL,
            PRIMARY KEY (user_id, key)
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS macro_data (
            key TEXT PRIMARY KEY,
            value DOUBLE PRECISION,
            previous_value DOUBLE PRECISION,
            change_pct DOUBLE PRECISION,
            updated_at TEXT NOT NULL
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            user_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            added_at TEXT NOT NULL,
            PRIMARY KEY (user_id, symbol)
        )
    """)
    db.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_user ON watchlist(user_id)")

    db.execute("""
        CREATE TABLE IF NOT EXISTS discovered_tickers (
            symbol TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            sector TEXT DEFAULT 'Unknown',
            index_name TEXT DEFAULT 'EGX',
            added_at TEXT NOT NULL
        )
    """)

    # Fundamentals from the TradingView scanner (see core/pe_fetch.py). `symbol`
    # is the exact ticker as returned by the feed — no name matching involved.
    # Null semantics: pe_ratio IS NULL means no trailing P/E (usually a
    # loss-maker); dividend_yield = 0.0 is REAL data meaning "pays nothing",
    # and only NULL means unknown; loss_making comes from diluted EPS, because
    # this source reports null rather than a negative P/E.
    db.execute("""
        CREATE TABLE IF NOT EXISTS pe_data (
            symbol TEXT PRIMARY KEY,
            company_name TEXT,
            pe_ratio DOUBLE PRECISION,
            dividend_yield DOUBLE PRECISION,
            updated_at TEXT NOT NULL
        )
    """)
    # Additive and idempotent, matching the rest of init_db — there is no
    # migration framework, so new columns land on the next cold start.
    db.execute("ALTER TABLE pe_data ADD COLUMN IF NOT EXISTS loss_making BOOLEAN")
    db.execute("CREATE INDEX IF NOT EXISTS idx_pe_data_updated ON pe_data(updated_at)")

    # Append-only change log of the PRICE-INDEPENDENT fundamentals.
    #
    # `pe_data` is a current-value read model: every refresh overwrites it, so
    # yesterday's numbers are destroyed nightly and no point-in-time question
    # can ever be answered. Without this table the valuation bands can never be
    # validated, because scoring a stock in the past with today's P/E is
    # look-ahead bias severe enough to manufacture any result you like.
    #
    # It stores EPS / DPS / book value rather than P/E, P/B and dividend yield.
    # Those three ratios all divide by PRICE, so they move every single day and
    # a log of them would be ~99% price noise. Verified against the live feed:
    # close / eps_ttm reproduces the reported P/E exactly. The ratio at any past
    # date is therefore reconstructable as (historical close from egxpy) / (the
    # fundamental in force on that date), which is both smaller to store and
    # more correct than logging the ratio itself.
    #
    # Rows are appended ONLY when a fundamental actually changes — these move
    # quarterly, so this stays small. `observed_at` is when WE saw the change,
    # not when the company reported it; the cron cadence bounds that lag to a
    # day. Point-in-time read: latest row for the symbol with observed_at <= X.
    db.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals_history (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            eps_ttm DOUBLE PRECISION,
            dps_annual DOUBLE PRECISION,
            book_value_per_share DOUBLE PRECISION,
            loss_making BOOLEAN,
            close_at_observation DOUBLE PRECISION
        )
    """)
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_fundamentals_history_symbol_time "
        "ON fundamentals_history(symbol, observed_at DESC)"
    )

    # Market-condition readings (see core/regime.py). Append-only. The reading
    # averages whatever the dashboard has already scored, so coverage varies
    # through the day — a persisted last-known-good is what lets the card show
    # this morning's reading instead of "no data" on a cold cache.
    db.execute("""
        CREATE TABLE IF NOT EXISTS market_regime (
            id BIGSERIAL PRIMARY KEY,
            observed_at TEXT NOT NULL,
            mean_score DOUBLE PRECISION,
            n_symbols INTEGER,
            band TEXT
        )
    """)
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_market_regime_time "
        "ON market_regime(observed_at DESC)"
    )

    # Per-symbol risk measurements, refreshed in CHUNKS by
    # POST /api/cron/risk_snapshot and ranked cross-sectionally at READ time.
    #
    # A current-value read model keyed by symbol, deliberately: there is no run
    # to finalize and no cursor state that can corrupt, so a half-finished
    # refresh still yields sensible percentiles because every row carries its
    # own most recent measurement. `measured_at` is per row, which is what lets
    # the read path report how stale its thinnest corner is instead of
    # presenting a partly-yesterday snapshot as today's.
    db.execute("""
        CREATE TABLE IF NOT EXISTS risk_snapshot (
            symbol TEXT PRIMARY KEY,
            measured_at TEXT NOT NULL,
            sigma_63_ann_pct DOUBLE PRECISION,
            sigma_ewma_ann_pct DOUBLE PRECISION,
            beta DOUBLE PRECISION,
            turnover_egp DOUBLE PRECISION,
            traded_share DOUBLE PRECISION,
            last_price DOUBLE PRECISION,
            tradeable BOOLEAN
        )
    """)
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_risk_snapshot_time "
        "ON risk_snapshot(measured_at DESC)"
    )

    # Twenty years of annual fundamentals, from the *_fy_h history arrays the
    # nightly scanner call already has access to. This is what makes a
    # fundamental factor testable at all: pe_data is a current-value snapshot
    # every refresh destroys, and fundamentals_history only starts 2026-08-25.
    #
    # first_usable_date = fiscal year end + 120 days, and it is the look-ahead
    # guard, not decoration. NOTHING may read a row before that date; go through
    # core/fundamentals_annual.get_annual_asof, which enforces it.
    db.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals_annual (
            symbol TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            eps_diluted DOUBLE PRECISION,
            dps DOUBLE PRECISION,
            net_income DOUBLE PRECISION,
            revenue DOUBLE PRECISION,
            total_assets DOUBLE PRECISION,
            gross_profit DOUBLE PRECISION,
            total_debt DOUBLE PRECISION,
            first_usable_date TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (symbol, fiscal_year)
        )
    """)
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_fundamentals_annual_usable "
        "ON fundamentals_annual(first_usable_date)"
    )

    # Dated macro observations. `released_at` is the load-bearing column, not
    # `observed_period`: a macro bar's timestamp is its REFERENCE period, and
    # August's inflation is not knowable in August. Reads go through
    # core/macro_series.get_macro_at, which filters on release.
    db.execute("""
        CREATE TABLE IF NOT EXISTS macro_series (
            series_code TEXT NOT NULL,
            observed_period TEXT NOT NULL,
            value DOUBLE PRECISION,
            released_at TEXT NOT NULL,
            source TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (series_code, observed_period)
        )
    """)
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_macro_series_released "
        "ON macro_series(series_code, released_at)"
    )

    # Columns added after pe_data shipped. No migration framework here — every
    # statement in init_db is idempotent, so these land on the next cold start.
    # market_cap is the one that matters: it is the missing size control on the
    # low-volatility result, the single open caveat on the app's best finding.
    for column, coltype in (
        ("market_cap", "DOUBLE PRECISION"),
        ("shares_outstanding", "DOUBLE PRECISION"),
        ("beta_1y", "DOUBLE PRECISION"),
        ("value_traded_egp", "DOUBLE PRECISION"),
    ):
        db.execute(
            f"ALTER TABLE pe_data ADD COLUMN IF NOT EXISTS {column} {coltype}"
        )

    for key in ("pe_last_successful_fetch", "pe_last_attempt_status"):
        db.execute(
            "INSERT INTO settings (key, value) VALUES (%s, '') ON CONFLICT (key) DO NOTHING",
            (key,),
        )

    db.execute(
        "INSERT INTO settings (key, value) VALUES ('currency', 'EGP') "
        "ON CONFLICT (key) DO NOTHING"
    )
    db.execute(
        "INSERT INTO settings (key, value) VALUES ('risk_free_rate', %s) "
        "ON CONFLICT (key) DO NOTHING",
        (str(DEFAULT_RISK_FREE_RATE_PCT),),
    )
    # The seed above is DO NOTHING, so an existing deployment would have kept the
    # stale 25% forever — and that number is the Sharpe hurdle, 13% of the
    # composite via score_risk_adjusted, and the bar realized trades are graded
    # against. Correct it, but ONLY where it still holds the old default, so an
    # admin who deliberately set a rate is not overwritten. Idempotent: once the
    # row moves off the stale value this matches nothing.
    db.execute(
        "UPDATE settings SET value = %s "
        "WHERE key = 'risk_free_rate' AND value = %s",
        (str(DEFAULT_RISK_FREE_RATE_PCT), str(STALE_RISK_FREE_RATE_PCT)),
    )

    # "Beginner Safe" composite weight defaults — ON CONFLICT keeps existing
    # DBs untouched.
    for key, default in (
        ("weight_trend", "18"),
        ("weight_momentum", "15"),
        ("weight_volume", "12"),
        ("weight_volatility", "10"),
        ("weight_divergence", "8"),
        ("weight_quality", "12"),
        ("weight_risk_adjusted", "13"),
        ("weight_relative_strength", "12"),
    ):
        db.execute(
            "INSERT INTO settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING",
            (key, default),
        )


def get_db():
    """Return a shared DB wrapper; initialize schema exactly once per process."""
    global _initialized
    db = _DB(_get_pool())
    if not _initialized:
        init_db(db)
        _initialized = True
    return db
