"""The real-Postgres tests drop tables, so they must only run against a local DB."""

from pg_guard import skip_reason


def test_unset_url_skips():
    assert skip_reason(None) == "TEST_DATABASE_URL not set"
    assert skip_reason("") == "TEST_DATABASE_URL not set"


def test_local_hosts_run():
    for url in ("postgresql://postgres:pw@localhost:55433/postgres",
                "postgresql://postgres:pw@127.0.0.1:5432/postgres",
                "postgresql://postgres:pw@[::1]:5432/postgres"):
        assert skip_reason(url) is None, url


def test_remote_host_skips():
    reason = skip_reason("postgresql://u:p@ep-cool-1.eu-central-1.aws.neon.tech/neondb")
    assert reason and "not local" in reason


def test_remote_host_runs_only_with_explicit_override():
    url = "postgresql://u:p@db.example.com/prod"
    assert skip_reason(url, "1") is None
    assert skip_reason(url, "true") is not None
