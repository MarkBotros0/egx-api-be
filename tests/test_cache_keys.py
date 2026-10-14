import ast
import pathlib

import pytest

from app.core.cache import make_key


def test_key_is_prefixed_by_market():
    assert make_key("EG", "ohlcv", "COMI") == "EG:ohlcv:COMI"


def test_unknown_market_is_refused():
    with pytest.raises(ValueError):
        make_key("ohlcv", "COMI")


def test_no_call_site_passes_a_literal_first_argument():
    """A literal first arg ('ohlcv', 'EG') means the market was hardcoded or forgotten."""
    bad = []
    for path in pathlib.Path("app").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) == "make_key":
                if node.args and isinstance(node.args[0], ast.Constant):
                    bad.append(f"{path}:{node.lineno}")
    assert bad == []
