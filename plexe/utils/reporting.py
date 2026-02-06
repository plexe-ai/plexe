"""
Utilities for saving agent reports to disk.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from plexe.constants import DirNames

logger = logging.getLogger(__name__)


def save_report(work_dir: Path, report_name: str, content: Any) -> Path:
    """
    Save agent report to workdir/DirNames.BUILD_DIR/reports/ as YAML.

    Automatically sanitizes numpy types and other non-native types for clean serialization.

    Args:
        work_dir: Working directory for this build
        report_name: Name of report (e.g., "statistical_analysis", "task_analysis")
        content: Report content (dict, list, or simple value)

    Returns:
        Path to saved report file
    """
    reports_dir = work_dir / DirNames.BUILD_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / f"{report_name}.yaml"

    # Sanitize content to ensure clean YAML serialization
    sanitized_content = _convert_to_native_types(content)

    with open(report_path, "w") as f:
        yaml.dump(sanitized_content, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info(f"Saved report: {report_path}")

    return report_path


def _convert_to_native_types(obj):
    """
    Recursively convert numpy types and other non-native types to Python native types.

    Ensures clean YAML/JSON serialization by converting:
    - numpy scalars (np.float64, np.int64) → Python float/int
    - numpy arrays → Python lists
    - Recursively processes dicts and lists

    Args:
        obj: Object to convert (can be dict, list, numpy type, or primitive)

    Returns:
        Object with all numpy types converted to Python primitives
    """
    try:
        # Handle NumPy types (check specific types before generic)
        if isinstance(obj, np.ndarray):
            return _convert_to_native_types(obj.tolist())
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.generic):
            return _convert_to_native_types(obj.item())
        # Recursively handle collections
        elif isinstance(obj, dict):
            return {k: _convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, list | tuple):
            return type(obj)(_convert_to_native_types(item) for item in obj)
        # Native types pass through
        elif isinstance(obj, str | int | float | bool | type(None)):
            return obj
        # Unsupported types trigger fallback
        else:
            raise TypeError(f"Unsupported type: {type(obj).__name__}")
    except Exception as e:
        # Fallback: double-serialization trick converts everything to native types
        logger.warning(f"Failed to convert {type(obj).__name__} to native types: {e}. Using string fallback.")
        return json.loads(json.dumps(obj, skipkeys=True, default=str))
