"""
Template for sklearn feature engineering pipeline.

This is an EXAMPLE showing possible patterns and structures. Your pipeline does NOT need to
follow this exact structure - adapt it to your specific task and data.

REQUIREMENTS (must follow):
- Code must define a variable named 'pipeline'
- 'pipeline' must be an sklearn.pipeline.Pipeline object
- Include ALL necessary imports in your code

RECOMMENDATIONS (optional):
- Use ColumnTransformer to handle different feature types
- Handle missing values appropriately
- Use handle_unknown='ignore' for encoders
- Set sparse_output=False for compatibility
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# TODO: Add any additional imports needed for your transformations


# TODO: Define your feature engineering pipeline below.
# ============================================
# EXAMPLE 1: Basic ColumnTransformer Pattern
# ============================================
# This is a common pattern but NOT required

numeric_features = ["age", "fare"]
categorical_features = ["sex", "embarked"]

numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

categorical_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    [("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, categorical_features)]
)

pipeline = Pipeline([("preprocessor", preprocessor)])

# ============================================
# EXAMPLE 2: Simple Single-Step Pipeline
# ============================================
# Sometimes simpler is better:
#
# from sklearn.preprocessing import StandardScaler
# pipeline = Pipeline([
#     ('scaler', StandardScaler())
# ])

# ============================================
# EXAMPLE 3: Custom Logic with FunctionTransformer
# ============================================
# For custom transformations, use FunctionTransformer:
#
# from sklearn.preprocessing import FunctionTransformer
# import numpy as np
# import pandas as pd
#
# def log_transform_columns(X):
#     """Apply log transform to specific columns."""
#     X_copy = X.copy()
#     X_copy['age'] = np.log1p(X_copy['age'])
#     X_copy['fare'] = np.log1p(X_copy['fare'])
#     return X_copy
#
# def clip_outliers(X):
#     """Clip values to 3 standard deviations."""
#     return np.clip(X, -3, 3)
#
# pipeline = Pipeline([
#     ('log_transform', FunctionTransformer(log_transform_columns)),
#     ('scaler', StandardScaler()),
#     ('clip', FunctionTransformer(clip_outliers))
# ])
#
# IMPORTANT: Define custom functions INSIDE your code string!
# Do NOT use custom transformer classes (BaseEstimator/TransformerMixin)

# ============================================
# TIPS
# ============================================
# - Keep it simple when possible
# - Test transformations make sense for your data
# - Consider feature interactions with PolynomialFeatures
# - Use feature selection (VarianceThreshold, SelectKBest) if needed
# - Always exclude target columns from the pipeline
