"""
Google / Apple ID-token verification with a locally generated RSA key and a
stub JWKS client — no network.

Run from egx-api-be:  python -m pytest tests/test_oauth.py -v
"""

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from app.core import oauth

KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class StubJWKS:
    def __init__(self, key):
        self.key = key.public_key()

    def get_signing_key_from_jwt(self, token):
        return type("K", (), {"key": self.key})()


def _token(claims, key=KEY):
    base = {"iat": int(time.time()), "exp": int(time.time()) + 600}
    return jwt.encode({**base, **claims}, key, algorithm="RS256")


@pytest.fixture(autouse=True)
def ids(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_IDS", "web-id, android-id")
    monkeypatch.setenv("APPLE_CLIENT_IDS", "com.markbotros.borsacompass,host.exp.Exponent")


GOOD_GOOGLE = {"iss": "https://accounts.google.com", "aud": "android-id", "sub": "g-1",
               "email": "Mark@Gmail.com", "email_verified": True, "name": "Mark B"}
GOOD_APPLE = {"iss": "https://appleid.apple.com", "aud": "com.markbotros.borsacompass",
              "sub": "a-1", "email": "x@privaterelay.appleid.com"}


def test_google_happy_path_normalises_email():
    ident = oauth.verify_google(_token(GOOD_GOOGLE), jwks=StubJWKS(KEY))
    assert ident == oauth.Identity(sub="g-1", email="mark@gmail.com", name="Mark B")


def test_google_accepts_the_bare_issuer_too():
    ident = oauth.verify_google(_token({**GOOD_GOOGLE, "iss": "accounts.google.com"}), jwks=StubJWKS(KEY))
    assert ident.sub == "g-1"


@pytest.mark.parametrize("bad", [
    {"aud": "someone-else"},
    {"iss": "https://evil.example"},
    {"exp": int(time.time()) - 10},
    {"email_verified": False},
    {"email": None},
])
def test_google_rejects(bad):
    with pytest.raises(HTTPException) as e:
        oauth.verify_google(_token({**GOOD_GOOGLE, **bad}), jwks=StubJWKS(KEY))
    assert e.value.status_code == 401


def test_google_rejects_a_token_signed_by_another_key():
    with pytest.raises(HTTPException):
        oauth.verify_google(_token(GOOD_GOOGLE, key=OTHER), jwks=StubJWKS(KEY))


def test_google_not_configured_is_503(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_IDS", "")
    with pytest.raises(HTTPException) as e:
        oauth.verify_google(_token(GOOD_GOOGLE), jwks=StubJWKS(KEY))
    assert e.value.status_code == 503


def test_apple_happy_path_and_missing_email():
    assert oauth.verify_apple(_token(GOOD_APPLE), jwks=StubJWKS(KEY)).email == "x@privaterelay.appleid.com"
    no_email = {k: v for k, v in GOOD_APPLE.items() if k != "email"}
    assert oauth.verify_apple(_token(no_email), jwks=StubJWKS(KEY)).email is None


def test_apple_accepts_expo_go_audience():
    assert oauth.verify_apple(_token({**GOOD_APPLE, "aud": "host.exp.Exponent"}), jwks=StubJWKS(KEY)).sub == "a-1"


@pytest.mark.parametrize("bad", [{"aud": "com.other.app"}, {"iss": "https://accounts.google.com"}])
def test_apple_rejects(bad):
    with pytest.raises(HTTPException):
        oauth.verify_apple(_token({**GOOD_APPLE, **bad}), jwks=StubJWKS(KEY))


def test_apple_exchange_is_skipped_when_not_configured(monkeypatch):
    for v in ("APPLE_TEAM_ID", "APPLE_KEY_ID", "APPLE_PRIVATE_KEY"):
        monkeypatch.delenv(v, raising=False)
    assert oauth.exchange_apple_code("code") is None


def test_revoke_never_raises(monkeypatch):
    monkeypatch.setattr(oauth, "_apple_post", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    monkeypatch.setenv("APPLE_TEAM_ID", "T"); monkeypatch.setenv("APPLE_KEY_ID", "K")
    monkeypatch.setenv("APPLE_PRIVATE_KEY", "not-a-key")
    oauth.revoke_apple_token("rt")   # no exception
