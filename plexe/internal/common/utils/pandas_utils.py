import pandas as pd


def convert_dtype_to_python(dtype, sample_values=None) -> str:
    """
    Convert a Pandas dtype to a Python type.

    :param dtype: The Pandas dtype to convert.
    :param sample_values: Optional sample values to help detect list types in object columns.
    :return: The corresponding Python type as string.
    """
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    elif pd.api.types.is_numeric_dtype(dtype):
        if pd.api.types.is_integer_dtype(dtype):
            return "int"
        else:
            return "float"
    elif dtype == "object" and sample_values is not None:
        # Check if object column contains lists by sampling values
        for val in sample_values:
            if pd.notna(val) and isinstance(val, list):
                # Detect element type from list contents using pandas-safe methods
                if not val:  # Empty list
                    continue
                first_elem = val[0]
                if pd.api.types.is_bool_dtype(type(first_elem)) or isinstance(first_elem, (bool, pd.BooleanDtype)):
                    return "List[bool]"
                elif pd.api.types.is_integer_dtype(type(first_elem)):
                    return "List[int]"
                elif pd.api.types.is_float_dtype(type(first_elem)):
                    return "List[float]"
                else:
                    return "List[str]"
        return "str"
    else:
        return "str"
