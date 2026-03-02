"""Shared test fixtures for plexe tests."""

import pytest


@pytest.fixture
def synthetic_parquet_classification(tmp_path):
    """Create a 200-row binary classification parquet with 2 row groups."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    np.random.seed(42)
    n_rows = 200
    n_features = 5

    X = np.random.randn(n_rows, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y

    path = tmp_path / "classification.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, row_group_size=100)

    return {
        "path": str(path),
        "target_column": "target",
        "n_rows": n_rows,
        "n_features": n_features,
        "n_classes": 2,
        "task_type": "binary_classification",
    }


@pytest.fixture
def synthetic_parquet_regression(tmp_path):
    """Create a 200-row regression parquet with 2 row groups."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    np.random.seed(42)
    n_rows = 200
    n_features = 5

    X = np.random.randn(n_rows, n_features).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1] + np.random.randn(n_rows) * 0.1).astype(np.float32)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y

    path = tmp_path / "regression.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, row_group_size=100)

    return {
        "path": str(path),
        "target_column": "target",
        "n_rows": n_rows,
        "n_features": n_features,
        "task_type": "regression",
    }
