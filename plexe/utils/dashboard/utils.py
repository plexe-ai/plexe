"""Utility functions for dashboard data loading."""

import json
from pathlib import Path

import pandas as pd
import yaml

from plexe.constants import DirNames


def load_report(exp_path: Path, report_name: str) -> dict | None:
    """Load YAML report from DirNames.BUILD_DIR/reports/."""
    report_path = exp_path / DirNames.BUILD_DIR / "reports" / f"{report_name}.yaml"
    if not report_path.exists():
        return None

    with open(report_path) as f:
        return yaml.safe_load(f)


def load_code_file(file_path: Path) -> str | None:
    """Load Python code file."""
    if not file_path.exists():
        return None

    return file_path.read_text()


def load_parquet_sample(uri: str, limit: int = 10) -> pd.DataFrame | None:
    """Load first N rows from parquet file."""
    try:
        return pd.read_parquet(uri).head(limit)
    except Exception:
        return None


def get_parquet_row_count(uri: str) -> int | None:
    """Get row count from parquet file."""
    try:
        return len(pd.read_parquet(uri))
    except Exception:
        return None


def load_json_file(file_path: Path) -> dict | None:
    """Load JSON file."""
    if not file_path.exists():
        return None

    with open(file_path) as f:
        return json.load(f)
