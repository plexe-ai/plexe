import pandas as pd


def convert_dtype_to_python(dtype) -> type:
    """
    Convert a Pandas dtype to a Python type.

    :param dtype: The Pandas dtype to convert.
    :return: The corresponding Python type.
    """
    if pd.api.types.is_bool_dtype(dtype):
        return bool
    elif pd.api.types.is_numeric_dtype(dtype):
        if pd.api.types.is_integer_dtype(dtype):
            return int
        else:
            return float
    else:
        return str
