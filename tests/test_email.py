"""
SMTP mailer + templates. smtplib is replaced with a recorder; nothing leaves
the machine.

Run from egx-api-be:  python -m pytest tests/test_email.py -v
"""

import ssl

import pytest

from app.core import email as email_mod
from app.core.email import (
    EmailNotConfigured,
    EmailSendFailed,
    SmtpMailer,
    already_registered_message,
    invite_message,
    link,
    reset_message,
    verify_message,
)


class _FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.calls = []
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, context=None):
        self.calls.append("starttls")
        self.tls_context = context

    def login(self, user, pw):
        self.calls.append(("login", user, pw))

    def send_message(self, msg):
        self.calls.append(("send", msg))


@pytest.fixture
def smtp_env(monkeypatch):
    monkeypatch.setenv("SMTP_USER", "me@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-pw")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("SMTP_PORT", raising=False)
    monkeypatch.delenv("SMTP_FROM", raising=False)
    _FakeSMTP.instances.clear()
    monkeypatch.setattr(email_mod.smtplib, "SMTP", _FakeSMTP)


def test_defaults_to_gmail_starttls(smtp_env):
    SmtpMailer.from_env().send("to@x.com", "Subj", "plain", "<p>html</p>")
    s = _FakeSMTP.instances[-1]
    assert (s.host, s.port) == ("smtp.gmail.com", 587)
    assert s.calls[0] == "starttls"
    assert s.tls_context is not None
    assert s.tls_context.verify_mode == ssl.CERT_REQUIRED
    assert s.tls_context.check_hostname is True
    assert s.calls[1] == ("login", "me@gmail.com", "app-pw")
    msg = s.calls[2][1]
    assert msg["To"] == "to@x.com"
    assert msg["From"] == "Borsa Compass <me@gmail.com>"
    assert msg["Subject"] == "Subj"
    assert msg.is_multipart()


def test_missing_credentials_is_503(monkeypatch):
    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    with pytest.raises(EmailNotConfigured) as e:
        SmtpMailer.from_env()
    assert e.value.status_code == 503


def test_non_numeric_port_is_503_not_a_crash(monkeypatch):
    monkeypatch.setenv("SMTP_USER", "me@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-pw")
    monkeypatch.setenv("SMTP_PORT", "abc")
    with pytest.raises(EmailNotConfigured) as e:
        SmtpMailer.from_env()
    assert e.value.status_code == 503


def test_smtp_errors_become_email_send_failed(smtp_env, monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(email_mod.smtplib, "SMTP", boom)
    with pytest.raises(EmailSendFailed):
        SmtpMailer.from_env().send("to@x.com", "s", "t", "h")


def test_links_use_fe_base_url_without_double_slash(monkeypatch):
    monkeypatch.setenv("FE_BASE_URL", "https://app.example.com/")
    assert link("/set-password", "abc") == "https://app.example.com/set-password?token=abc"


def test_links_require_fe_base_url(monkeypatch):
    monkeypatch.delenv("FE_BASE_URL", raising=False)
    with pytest.raises(EmailNotConfigured):
        link("/verify-email", "abc")


@pytest.mark.parametrize("build", [
    lambda: verify_message("Mark", "https://x/verify-email?token=t"),
    lambda: reset_message("", "https://x/set-password?token=t"),
    lambda: invite_message("Sara", "Mark", "https://x/set-password?token=t"),
    lambda: already_registered_message("", "https://x/login", "https://x/forgot-password"),
])
def test_every_template_has_subject_text_and_html_with_the_link(build):
    subject, text, html = build()
    assert subject and "Borsa Compass" in (subject + text)
    assert "https://x/" in text and "https://x/" in html


def test_html_escapes_names():
    _, _, html = invite_message("<b>x</b>", "<script>", "https://x/set-password?token=t")
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_greeting_falls_back_when_name_is_empty():
    _, text, _ = verify_message("", "https://x/verify-email?token=t")
    assert text.startswith("Hi,")
