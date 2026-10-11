"""
Outgoing email over SMTP (a Gmail app password in production).

Env: SMTP_HOST (default smtp.gmail.com), SMTP_PORT (587), SMTP_USER,
SMTP_PASSWORD, SMTP_FROM (default "Borsa Compass <SMTP_USER>"), and
FE_BASE_URL — every emailed link points at the web app, which hosts
/verify-email and /set-password for both clients.

Routes that must send depend on get_mailer(); a missing config is a 503,
never a silent success. Tests replace the mailer with
tests/account_fakes.OutboxMailer.
"""

import html as _html
import os
import smtplib
from email.message import EmailMessage
from typing import Protocol

from fastapi import HTTPException

APP_NAME = "Borsa Compass"


class EmailNotConfigured(HTTPException):
    def __init__(self):
        super().__init__(status_code=503, detail="Email is not configured")


class EmailSendFailed(Exception):
    """SMTP refused or failed; routes turn this into 502."""


class Mailer(Protocol):
    def send(self, to: str, subject: str, text: str, html: str) -> None: ...


class SmtpMailer:
    def __init__(self, host: str, port: int, user: str, password: str, sender: str):
        self.host, self.port = host, port
        self.user, self.password, self.sender = user, password, sender

    @classmethod
    def from_env(cls) -> "SmtpMailer":
        user = os.environ.get("SMTP_USER", "").strip()
        password = os.environ.get("SMTP_PASSWORD", "").strip()
        if not user or not password:
            raise EmailNotConfigured()
        return cls(
            host=os.environ.get("SMTP_HOST", "").strip() or "smtp.gmail.com",
            port=int(os.environ.get("SMTP_PORT", "").strip() or 587),
            user=user,
            password=password,
            sender=os.environ.get("SMTP_FROM", "").strip() or f"{APP_NAME} <{user}>",
        )

    def send(self, to: str, subject: str, text: str, html: str) -> None:
        msg = EmailMessage()
        msg["From"] = self.sender
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(text)
        msg.add_alternative(html, subtype="html")
        try:
            with smtplib.SMTP(self.host, self.port, timeout=15) as smtp:
                smtp.starttls()
                smtp.login(self.user, self.password)
                smtp.send_message(msg)
        except (smtplib.SMTPException, OSError) as e:
            print(f"[email] send to {to} failed: {e}")
            raise EmailSendFailed(str(e)) from e


def get_mailer() -> Mailer:
    """FastAPI dependency. Overridden in tests."""
    return SmtpMailer.from_env()


def _base_url() -> str:
    base = os.environ.get("FE_BASE_URL", "").strip().rstrip("/")
    if not base:
        raise EmailNotConfigured()
    return base


def link(path: str, token: str) -> str:
    return f"{_base_url()}{path}?token={token}"


def login_url() -> str:
    return f"{_base_url()}/login"


def forgot_url() -> str:
    return f"{_base_url()}/forgot-password"


def _greeting(name: str) -> str:
    return f"Hi {name}," if name else "Hi,"


def _render(name: str, paragraphs: list, button_label: str, url: str, footer: str):
    text = "\n\n".join([_greeting(name), *paragraphs, f"{button_label}: {url}", footer,
                        f"— {APP_NAME}"])
    esc = _html.escape
    body = "".join(f"<p>{esc(p)}</p>" for p in paragraphs)
    html = (
        f'<div style="font-family:Arial,sans-serif;font-size:15px;color:#111">'
        f"<p>{esc(_greeting(name))}</p>{body}"
        f'<p><a href="{esc(url)}" style="display:inline-block;padding:10px 18px;'
        f'background:#4488ff;color:#fff;border-radius:8px;text-decoration:none">'
        f"{esc(button_label)}</a></p>"
        f'<p style="color:#666;font-size:13px">{esc(footer)}<br>'
        f'If the button does not work, open this link: {esc(url)}</p>'
        f"<p>— {APP_NAME}</p></div>"
    )
    return text, html


def verify_message(name: str, url: str):
    text, html = _render(
        name, [f"Confirm your email address to finish creating your {APP_NAME} account."],
        "Confirm email", url, "This link expires in 24 hours. If you did not sign up, ignore this email.")
    return f"Confirm your {APP_NAME} email", text, html


def reset_message(name: str, url: str):
    text, html = _render(
        name, ["Someone asked to reset the password for this account."],
        "Choose a new password", url,
        "This link expires in 1 hour. If it wasn't you, ignore this email — your password stays the same.")
    return f"Reset your {APP_NAME} password", text, html


def invite_message(name: str, inviter: str, url: str):
    who = inviter or "An admin"
    text, html = _render(
        name, [f"{who} invited you to {APP_NAME}. Choose a password to activate your account. "
               "You can also sign in with Google using this email address."],
        "Set your password", url, "This invitation expires in 7 days.")
    return f"{who} invited you to {APP_NAME}", text, html


def already_registered_message(name: str, login: str, reset: str):
    text, html = _render(
        name, ["Someone tried to create an account with this email, but you already have one.",
               f"If it was you, sign in here: {login}"],
        "Reset your password", reset, "If it wasn't you, ignore this email.")
    return f"You already have a {APP_NAME} account", text, html
