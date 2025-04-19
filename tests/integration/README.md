# Plexe Integration Tests


## 1. Overview
This directory contains integration tests validating the end-to-end functionality of Plexe across various ML tasks.


## 2. Test Suite
### 2.1 Classification
- **Binary Classification** (`test_binary_classification.py`): Heart disease prediction; tests model lifecycle.
- **Multiclass Classification** (`test_multiclass_classification.py`): Sentiment analysis; tests synthetic data generation.

### 2.2 Regression & Forecasting
- **Regression** (`test_regression.py`): House price prediction; tests schema specification and inference.
- **Time Series Forecasting** (`test_time_series.py`): Sales prediction; tests multi-feature temporal forecasting.

### 2.3 Other ML Tasks
- **Recommendation** (`test_recommendation.py`): Product recommendations; tests list-based output and cross-selling.
- **Customer Churn** (`test_customer_churn.py`): Churn prediction; tests probability outputs and schema validation.
- **Schema Validation** (`test_schema_validation.py`): Validates complex schemas using Pydantic.


## 3. Execution
### 3.1 Prerequisites
- Plexe installed with development dependencies.
- `OPENAI_API_KEY` set as an environment variable.

### 3.2 Running Tests

#### 3.2.1 Single Test
```bash
poetry run pytest tests/integration/test_<name>.py -v
```
Example:
```bash
poetry run pytest tests/integration/test_binary_classification.py -v
```

#### 3.2.2 All Tests
```bash
poetry run pytest tests/integration/ -v
```

#### 3.2.3 Filtering Tests
```bash
poetry run pytest -k <test_name> -v
```
Example:
```bash
poetry run pytest tests/integration/test_binary_classification.py::test_heart_disease_classification -v
```

### 3.3 Optimization Tips
- Run tests individually to reduce execution time.
- Set `max_iterations` to 2-3 and timeouts to ~10 minutes.
- Use small synthetic datasets (~30-60 samples).


## 4. Adding New Tests
- Scope tests to a single model type or feature.
- Ensure runtime-generated synthetic data.
- Validate model training, inference, saving/loading.
- Confirm schema compliance and expected outputs.
- Use `openai/gpt-4o` for all tests.

