"""
Google and Apple sign-in: verify the provider's signed ID token ourselves
(PyJWT + the provider's JWKS). No SDK, no Firebase.

Env:
  GOOGLE_CLIENT_IDS  comma-separated OAuth client ids (web, Android, iOS)
  APPLE_CLIENT_IDS   comma-separated audiences: the bundle id, plus
                     host.exp.Exponent so Expo Go works in development
  APPLE_TEAM_ID, APPLE_KEY_ID, APPLE_PRIVATE_KEY (.p8 contents)
                     optional; enable the code→refresh-token exchange that
                     lets account deletion revoke Apple's grant
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import jwt
from fastapi import HTTPException

GOOGLE_JWKS = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = ("https://accounts.google.com", "accounts.google.com")
APPLE_JWKS = "https://appleid.apple.com/auth/keys"
APPLE_ISSUER = "https://appleid.apple.com"
APPLE_TOKEN_URL = "https://appleid.apple.com/auth/token"
APPLE_REVOKE_URL = "https://appleid.apple.com/auth/revoke"


@dataclass(frozen=True)
class Identity:
    sub: str
    email: Optional[str]
    name: str


def _client_ids(env_var: str) -> list:
    return [c.strip() for c in os.environ.get(env_var, "").split(",") if c.strip()]


@lru_cache(maxsize=2)
def _jwks(url: str):
    return jwt.PyJWKClient(url, cache_keys=True, timeout=5)


def _decode(token: str, jwks, audiences: list, issuers, label: str) -> dict:
    try:
        key = jwks.get_signing_key_from_jwt(token).key
    except (jwt.PyJWKClientConnectionError, urllib.error.URLError, OSError):
        raise HTTPException(status_code=503, detail=f"{label} sign-in is temporarily unavailable")
    except jwt.PyJWKClientError as e:
        # Fetching the key set failed -> outage; an unknown kid is a bad token.
        if "Unable to find a signing key" in str(e):
            raise HTTPException(status_code=401, detail=f"{label} sign-in failed")
        raise HTTPException(status_code=503, detail=f"{label} sign-in is temporarily unavailable")
    except Exception:
        raise HTTPException(status_code=401, detail=f"{label} sign-in failed")
    try:
        claims = jwt.decode(token, key, algorithms=["RS256"], audience=audiences, leeway=60,
                            options={"require": ["exp", "iat", "sub", "iss", "aud"]})
    except Exception:
        raise HTTPException(status_code=401, detail=f"{label} sign-in failed")
    if claims.get("iss") not in issuers:
        raise HTTPException(status_code=401, detail=f"{label} sign-in failed")
    return claims


def verify_google(id_token: str, jwks=None) -> Identity:
    audiences = _client_ids("GOOGLE_CLIENT_IDS")
    if not audiences:
        raise HTTPException(status_code=503, detail="Google sign-in is not configured")
    claims = _decode(id_token, jwks or _jwks(GOOGLE_JWKS), audiences, GOOGLE_ISSUERS, "Google")
    email = (claims.get("email") or "").strip().lower()
    if not email or claims.get("email_verified") is not True:
        raise HTTPException(status_code=401, detail="Google sign-in failed")
    return Identity(sub=str(claims["sub"]), email=email, name=claims.get("name") or "")


def verify_apple(identity_token: str, jwks=None) -> Identity:
    audiences = _client_ids("APPLE_CLIENT_IDS")
    if not audiences:
        raise HTTPException(status_code=503, detail="Apple sign-in is not configured")
    claims = _decode(identity_token, jwks or _jwks(APPLE_JWKS), audiences, (APPLE_ISSUER,), "Apple")
    email = (claims.get("email") or "").strip().lower() or None
    if claims.get("email_verified") not in (True, "true"):
        email = None   # an unverified address must never link or create accounts
    # Apple never puts the name in the token; the app sends it (first sign-in only).
    return Identity(sub=str(claims["sub"]), email=email, name="")


def _apple_configured() -> bool:
    return all(os.environ.get(v, "").strip() for v in ("APPLE_TEAM_ID", "APPLE_KEY_ID", "APPLE_PRIVATE_KEY"))


def _apple_client_secret() -> str:
    now = int(time.time())
    key = os.environ["APPLE_PRIVATE_KEY"].replace("\\n", "\n")
    return jwt.encode(
        {"iss": os.environ["APPLE_TEAM_ID"], "iat": now, "exp": now + 300,
         "aud": APPLE_ISSUER, "sub": _client_ids("APPLE_CLIENT_IDS")[0]},
        key, algorithm="ES256", headers={"kid": os.environ["APPLE_KEY_ID"]},
    )


def _apple_post(url: str, fields: dict) -> dict:
    data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read()
    return json.loads(body) if body else {}


def exchange_apple_code(code: str) -> Optional[str]:
    """Trade the one-time authorization code for a refresh token, or None."""
    if not code or not _apple_configured() or not _client_ids("APPLE_CLIENT_IDS"):
        return None
    try:
        out = _apple_post(APPLE_TOKEN_URL, {
            "client_id": _client_ids("APPLE_CLIENT_IDS")[0],
            "client_secret": _apple_client_secret(),
            "code": code,
            "grant_type": "authorization_code",
        })
        return out.get("refresh_token")
    except Exception as e:
        print(f"[oauth] apple code exchange failed: {e}")
        return None


def revoke_apple_token(refresh_token: str) -> None:
    """Best-effort: account deletion must never fail because Apple is down."""
    if not refresh_token or not _apple_configured():
        return
    try:
        _apple_post(APPLE_REVOKE_URL, {
            "client_id": _client_ids("APPLE_CLIENT_IDS")[0],
            "client_secret": _apple_client_secret(),
            "token": refresh_token,
            "token_type_hint": "refresh_token",
        })
    except Exception as e:
        print(f"[oauth] apple revoke failed: {e}")
