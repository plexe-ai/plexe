"""Test that all production modules can be imported without errors."""

import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path


def test_all_modules_importable():
    """
    Import all production modules in the plexe/ package to catch import errors.

    This test will fail if:
    - Any module uses features not available in the current Python version
    - Any import fails for any reason (missing dependencies, syntax errors, etc.)

    Note: Dashboard/viz modules are excluded as they have optional dependencies.
    """
    parent_dir = Path(__file__).parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    import plexe

    failed = []
    success_count = 0
    missing_optional = set()
    if importlib.util.find_spec("catboost") is None:
        missing_optional.add("catboost")
    if importlib.util.find_spec("lightgbm") is None:
        missing_optional.add("lightgbm")
    if importlib.util.find_spec("torch") is None:
        missing_optional.add("torch")

    for _importer, modname, _ispkg in pkgutil.walk_packages(
        path=plexe.__path__,
        prefix="plexe.",
    ):
        # Skip dashboard/viz modules — optional dependencies (streamlit/plotly)
        if "dashboard" in modname or modname == "plexe.viz":
            continue
        if "catboost" in modname and "catboost" in missing_optional:
            continue
        if "lightgbm" in modname and "lightgbm" in missing_optional:
            continue
        if "pytorch" in modname and "torch" in missing_optional:
            continue
        if "torch" in modname and "torch" in missing_optional:
            continue

        try:
            importlib.import_module(modname)
            success_count += 1
        except Exception as e:
            failed.append(f"  {modname}: {type(e).__name__}: {e}")

    assert not failed, (
        f"Failed to import {len(failed)} module(s) on Python {sys.version_info.major}.{sys.version_info.minor}:\n"
        + "\n".join(failed)
        + f"\n\nSuccessfully imported {success_count} module(s)"
    )
