"""
Tag a headline with the EGX symbols it names.

TradingView stories arrive tagged (`relatedSymbols`); Arab Finance's do not, so
the only way to put one on a holding is to read its headline for a company
name. This module is that reader — no network, built once per process from
`data/egx_tickers.json` plus a short hand-kept alias list.

Three rules keep it from inventing links, which is worse than missing one (a
story on the wrong stock's page reads as a fact about that stock):

  1. **Whole words only.** "CIB" must not match inside "CIBC"; "Fawry's" still
     matches "Fawry" (an apostrophe is not a word character).
  2. **An alias that names two symbols names neither.** The ticker file has
     duplicates (GLCM and GTHE are both "Global Telecom Holding"), and
     guessing between them would tag a coin flip. A hand-kept alias overrides,
     because a person chose it.
  3. **The longest match wins.** "CIB Holdings" (CICH) contains "CIB" (COMI);
     a match lying inside a longer one is dropped, so a CICH story is not also
     filed under COMI.

Single words are only aliases when they are an abbreviation the ticker file
spells in capitals (CIB, SODIC, MOPCO) or a hand-kept entry (Fawry, Juhayna).
Trimming "Eastern Company" to "Eastern" or "Pioneer Holding" to "Pioneer"
would match ordinary English, so trimming stops at two words.

KNOWN LIMIT: `data/egx_tickers.json` is drifting (see CLAUDE.md), so a company
filed under a wrong or stale name is matched under that name. Tagging is a
convenience on top of the market feed, not a claim of completeness.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Optional

_TICKERS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "egx_tickers.json"
)

# Names the press actually uses that the ticker file does not spell out. Kept
# short on purpose: each entry is a judgement that the name is unambiguous.
MANUAL_ALIASES: dict[str, tuple[str, ...]] = {
    "COMI": ("CIB", "Commercial International Bank"),
    "ETEL": ("Telecom Egypt",),
    "FWRY": ("Fawry",),
    "HRHO": ("EFG Hermes", "EFG Holding"),
    "TMGH": ("TMG", "TMG Holding", "Talaat Moustafa Group", "Talaat Moustafa"),
    "OCDI": ("SODIC",),
    "PHDC": ("Palm Hills",),
    "JUFO": ("Juhayna",),
    "SWDY": ("Elsewedy Electric", "El Sewedy Electric"),
    "ABUK": ("Abu Qir Fertilizers",),
    "EFIH": ("e-finance",),
    "MASR": ("Madinet Masr",),
}

# Words a company name may end in that do not identify it.
_GENERIC_SUFFIXES = {"holding", "holdings", "company", "co", "group", "s.a.e", "sae"}

_ABBREVIATION = re.compile(r"^[A-Z][A-Z0-9&]{2,}$")


def _load_names(path: str = _TICKERS_PATH) -> list[tuple[str, str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception:
        return []
    out = []
    for row in rows if isinstance(rows, list) else []:
        symbol = (row.get("symbol") or "").strip().upper()
        name = (row.get("name") or "").strip()
        if symbol and name:
            out.append((symbol, name))
    return out


def derive_aliases(name: str) -> tuple[set[str], set[str]]:
    """
    (case-insensitive phrases, case-sensitive abbreviations) for one name.

    "Commercial International Bank (CIB)" -> ({"Commercial International Bank"}, {"CIB"})
    "EFG Hermes Holding"                  -> ({"EFG Hermes Holding", "EFG Hermes"}, set())
    """
    phrases: set[str] = set()
    abbreviations: set[str] = set()

    for inner in re.findall(r"\(([^)]*)\)", name):
        inner = inner.strip()
        if _ABBREVIATION.match(inner):
            abbreviations.add(inner)

    base = " ".join(re.sub(r"\([^)]*\)", " ", name).split())
    words = base.split()
    if len(words) >= 2:
        phrases.add(base)
    while len(words) > 2 and words[-1].lower().strip(".,") in _GENERIC_SUFFIXES:
        words = words[:-1]
        phrases.add(" ".join(words))
    return phrases, abbreviations


def _pattern(alias: str, case_sensitive: bool) -> re.Pattern:
    body = r"\s+".join(
        "(?:&|and)" if part == "&" else re.escape(part) for part in alias.split()
    )
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(rf"(?<!\w){body}(?!\w)", flags)


@lru_cache(maxsize=1)
def _index(path: str = _TICKERS_PATH) -> tuple[tuple[re.Pattern, str], ...]:
    """Every unambiguous alias, compiled, paired with its one symbol."""
    owners: dict[tuple[str, bool], set[str]] = {}
    for symbol, name in _load_names(path):
        phrases, abbreviations = derive_aliases(name)
        for p in phrases:
            owners.setdefault((p.lower(), False), set()).add(symbol)
        for a in abbreviations:
            owners.setdefault((a, True), set()).add(symbol)

    chosen: dict[tuple[str, bool], str] = {
        key: next(iter(symbols)) for key, symbols in owners.items() if len(symbols) == 1
    }
    # A hand-kept alias is a decision, so it overrides whatever the file implied
    # — including resolving an alias the file made ambiguous.
    for symbol, aliases in MANUAL_ALIASES.items():
        for alias in aliases:
            sensitive = bool(_ABBREVIATION.match(alias))
            chosen[(alias if sensitive else alias.lower(), sensitive)] = symbol

    return tuple(
        (_pattern(alias, sensitive), symbol) for (alias, sensitive), symbol in chosen.items()
    )


def tag_symbols(title: str, index: Optional[tuple] = None) -> list[str]:
    """The EGX symbols a headline names, sorted. [] when it names none."""
    if not title:
        return []
    hits: list[tuple[int, int, str]] = []
    for pattern, symbol in index if index is not None else _index():
        for m in pattern.finditer(title):
            hits.append((m.start(), m.end(), symbol))

    kept = {
        symbol
        for start, end, symbol in hits
        if not any(
            (s <= start and end <= e) and (e - s) > (end - start)
            for s, e, _ in hits
        )
    }
    return sorted(kept)
