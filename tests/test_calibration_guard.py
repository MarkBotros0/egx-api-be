"""
No fitted constant may live outside app/core/calibration/. The old global
names are listed here; any app/ module that defines, imports or reads one
fails the build. (Task 3 adds the liquidity table.)
"""
import ast
import pathlib

APP = pathlib.Path(__file__).resolve().parents[1] / "app"

FORBIDDEN = {
    "EGX_Z_CONE_60D", "ONE_SIGMA_COVERAGE_PCT", "CALIBRATION",
    "REGIME_WEAK_MAX", "REGIME_MIXED_MAX", "REGIME_ASSOCIATION_RHO",
    "REGIME_ASSOCIATION_T", "REGIME_ASSOCIATION_N",
    "REGIME_ASSOCIATION_SIGNIFICANT", "REGIME_PHASE_RHOS",
    "RISK_QUINTILES", "RISK_CALIBRATION",
}


def offences(names: set) -> list:
    found = []
    for path in sorted(APP.rglob("*.py")):
        if "vendor" in path.parts:
            continue
        rel = path.relative_to(APP)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name) and n.id in names:
                        found.append(f"{rel}:{node.lineno} defines {n.id}")
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in names:
                        found.append(f"{rel}:{node.lineno} imports {alias.name}")
            if isinstance(node, ast.Attribute) and node.attr in names:
                found.append(f"{rel}:{node.lineno} reads .{node.attr}")
    return found


def test_no_app_module_defines_or_imports_a_removed_global():
    assert offences(FORBIDDEN) == []
