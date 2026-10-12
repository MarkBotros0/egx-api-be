"""
ONE-OFF: delete every user and everything they own, then recreate `users`
in the email-account shape and seed AUTH_ADMINS.

    DATABASE_URL=... AUTH_ADMINS=mark3essam@gmail.com \
        python scripts/reset_users_for_email_auth.py

It prints row counts and asks you to type DELETE ALL USERS. There is no undo.

Rollout order (the same in the spec §9 and CLAUDE.md):
  1. set the backend env vars (FRONTEND_ORIGIN first entry = the production
     web origin, SMTP_*, AUTH_ADMINS, GOOGLE_CLIENT_IDS, APPLE_*) and the web's
     NEXT_PUBLIC_GOOGLE_CLIENT_ID;
  2. run THIS script against production, from the email-auth code;
  3. deploy the backend;
  4. deploy the web.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import db as db_mod  # noqa: E402
from app.core.accounts import USER_OWNED_TABLES  # noqa: E402
from app.core.auth import seed_users_from_env  # noqa: E402

CONFIRM = "DELETE ALL USERS"


def main() -> int:
    if not os.environ.get("DATABASE_URL"):
        print("DATABASE_URL is not set")
        return 2
    db_mod._initialized = True          # do not run init_db against the old table yet
    db = db_mod.get_db()

    tables = (*USER_OWNED_TABLES, "users")
    print("Rows that will be deleted:")
    for t in tables:
        exists = db.execute("SELECT to_regclass(%s)", (t,)).fetchone()[0]
        n = db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] if exists else 0
        print(f"  {t:22s} {n}")
    if input(f'Type "{CONFIRM}" to continue: ').strip() != CONFIRM:
        print("Aborted — nothing changed.")
        return 1

    with db.transaction() as tx:
        for t in USER_OWNED_TABLES:
            if tx.execute("SELECT to_regclass(%s)", (t,)).fetchone()[0]:
                tx.execute(f"DELETE FROM {t}")
        tx.execute("DROP TABLE IF EXISTS users")
    db_mod.init_db(db)
    seed_users_from_env(db)
    admins = db.execute("SELECT email, role FROM users").fetchall()
    print(f"Done. Users now: {admins}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
