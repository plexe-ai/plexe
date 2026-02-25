"""Unit tests for reporting utilities."""

import numpy as np

from plexe.utils.reporting import _convert_to_native_types


def test_convert_to_native_types_numpy_nested():
    """Numpy scalars/arrays should become native Python types."""
    data = {
        "a": np.int64(3),
        "b": np.float64(2.5),
        "c": np.array([1, 2]),
        "d": {"e": np.bool_(True)},
        "f": [np.float32(1.25), {"g": np.array([3, 4])}],
    }

    converted = _convert_to_native_types(data)

    assert converted["a"] == 3
    assert isinstance(converted["a"], int)
    assert converted["b"] == 2.5
    assert isinstance(converted["b"], float)
    assert converted["c"] == [1, 2]
    assert isinstance(converted["d"]["e"], bool)
    assert converted["f"][0] == 1.25
    assert converted["f"][1]["g"] == [3, 4]
