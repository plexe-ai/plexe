"""Unit tests for reporting utilities."""

import numpy as np
import yaml

from plexe.utils.reporting import save_report


def test_save_report_converts_numpy_types(tmp_path):
    """save_report should serialize numpy types to native Python values."""
    data = {
        "a": np.int64(3),
        "b": np.float64(2.5),
        "c": np.array([1, 2]),
        "d": {"e": np.bool_(True)},
        "f": [np.float32(1.25), {"g": np.array([3, 4])}],
    }

    report_path = save_report(tmp_path, "sample_report", data)
    loaded = yaml.safe_load(report_path.read_text())

    assert loaded["a"] == 3
    assert isinstance(loaded["a"], int)
    assert loaded["b"] == 2.5
    assert isinstance(loaded["b"], float)
    assert loaded["c"] == [1, 2]
    assert isinstance(loaded["d"]["e"], bool)
    assert loaded["f"][0] == 1.25
    assert loaded["f"][1]["g"] == [3, 4]
