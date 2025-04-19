# tests/utils/utils.py
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

from plexe.internal.models.entities.description import ModelDescription


def generate_heart_data(n_samples=30, random_seed=42):
    """Generate synthetic heart disease data for testing.

    The data follows the structure:
    - age: int (25-80)
    - gender: int (0=female, 1=male)
    - cp: int (chest pain type, 0-3)
    - trtbps: int (resting blood pressure, 90-200)
    - chol: int (cholesterol, 120-400)
    - fbs: int (fasting blood sugar > 120 mg/dl, 0-1)
    - restecg: int (resting ECG results, 0-2)
    - thalachh: int (maximum heart rate achieved, 70-220)
    - exng: int (exercise induced angina, 0-1)
    - oldpeak: float (ST depression induced by exercise, 0-6.0)
    - slp: int (slope of peak exercise ST segment, 0-2)
    - caa: int (number of major vessels, 0-4)
    - thall: int (thalassemia, 0-3)
    - output: int (presence of heart disease, 0-1)
    """
    np.random.seed(random_seed)

    # Generate features
    data = {
        "age": np.random.randint(25, 80, n_samples),
        "gender": np.random.randint(0, 2, n_samples),
        "cp": np.random.randint(0, 4, n_samples),
        "trtbps": np.random.randint(90, 200, n_samples),
        "chol": np.random.randint(120, 400, n_samples),
        "fbs": np.random.randint(0, 2, n_samples),
        "restecg": np.random.randint(0, 3, n_samples),
        "thalachh": np.random.randint(70, 220, n_samples),
        "exng": np.random.randint(0, 2, n_samples),
        "oldpeak": np.round(np.random.uniform(0, 6, n_samples), 1),
        "slp": np.random.randint(0, 3, n_samples),
        "caa": np.random.randint(0, 5, n_samples),
        "thall": np.random.randint(0, 4, n_samples),
    }

    # Generate target based on risk factors
    risk_factors = (
        (data["age"] > 60).astype(int) * 2  # Age over 60 is high risk
        + data["gender"]  # Being male slightly increases risk
        + (data["cp"] > 1).astype(int) * 2  # Higher chest pain types increase risk
        + (data["trtbps"] > 140).astype(int)  # High blood pressure
        + (data["chol"] > 250).astype(int)  # High cholesterol
        + data["fbs"]  # High fasting blood sugar
        + (data["thalachh"] < 120).astype(int) * 2  # Low max heart rate
        + data["exng"] * 2  # Exercise-induced angina
        + (data["oldpeak"] > 2).astype(int) * 2  # High ST depression
        + data["caa"]  # Number of major vessels
    )

    # Convert risk factors to binary output (threshold chosen to get roughly balanced classes)
    data["output"] = (risk_factors > 8).astype(int)

    return pd.DataFrame(data)


def generate_house_prices_data(n_samples=30, random_seed=42):
    """Generate synthetic house price data for regression testing.

    The data follows the structure:
    - area: int (square feet, 800-5000)
    - bedrooms: int (1-6)
    - bathrooms: int (1-5)
    - stories: int (1-4)
    - garage: int (0-3 cars)
    - garden: int (0=no, 1=yes)
    - fenced: int (0=no, 1=yes)
    - age: int (years, 0-100)
    - price: float (house price in thousands, 100-2000)
    """
    np.random.seed(random_seed)

    # Generate features
    data = {
        "area": np.random.randint(800, 5000, n_samples),
        "bedrooms": np.random.randint(1, 7, n_samples),
        "bathrooms": np.random.randint(1, 6, n_samples),
        "stories": np.random.randint(1, 5, n_samples),
        "garage": np.random.randint(0, 4, n_samples),
        "garden": np.random.randint(0, 2, n_samples),
        "fenced": np.random.randint(0, 2, n_samples),
        "age": np.random.randint(0, 101, n_samples),
    }

    # Generate price based on features
    # Base price
    price = 100 + np.random.normal(0, 20, n_samples)

    # Add impact of features
    price += data["area"] * 0.2  # Larger area increases price
    price += data["bedrooms"] * 25  # More bedrooms increase price
    price += data["bathrooms"] * 35  # More bathrooms increase price
    price += data["stories"] * 30  # More stories increase price
    price += data["garage"] * 40  # Garage increases price
    price += data["garden"] * 50  # Garden increases price
    price += data["fenced"] * 25  # Fenced yard increases price
    price -= data["age"] * 1.5  # Older houses decrease in price

    # Add some noise
    price += np.random.normal(0, 50, n_samples)

    # Ensure reasonable price range
    price = np.clip(price, 100, 2000)
    data["price"] = np.round(price, 2)

    return pd.DataFrame(data)


def generate_customer_churn_data(n_samples=30, random_seed=42):
    """Generate synthetic customer churn data for classification testing.

    The data follows the structure:
    - tenure: int (months, 0-100)
    - monthly_charges: float (dollars, 20-150)
    - total_charges: float (dollars, 0-10000)
    - contract_type: int (0=month-to-month, 1=one year, 2=two year)
    - payment_method: int (0=electronic check, 1=mailed check, 2=bank transfer, 3=credit card)
    - tech_support: int (0=no, 1=yes)
    - online_backup: int (0=no, 1=yes)
    - online_security: int (0=no, 1=yes)
    - churn: int (0=no, 1=yes)
    """
    np.random.seed(random_seed)

    # Generate features
    data = {
        "tenure": np.random.randint(1, 101, n_samples),
        "monthly_charges": np.round(np.random.uniform(20, 150, n_samples), 2),
        "contract_type": np.random.randint(0, 3, n_samples),
        "payment_method": np.random.randint(0, 4, n_samples),
        "tech_support": np.random.randint(0, 2, n_samples),
        "online_backup": np.random.randint(0, 2, n_samples),
        "online_security": np.random.randint(0, 2, n_samples),
    }

    # Calculate total charges based on tenure and monthly charges
    # Add some variance to simulate different starting points and promotional offers
    variance_factor = np.random.uniform(0.8, 1.2, n_samples)
    data["total_charges"] = np.round(data["tenure"] * data["monthly_charges"] * variance_factor, 2)

    # Generate churn based on risk factors
    risk_factors = (
        (data["tenure"] < 12).astype(int) * 3  # Low tenure is high risk
        + (data["monthly_charges"] > 100).astype(int) * 2  # High monthly charges
        + (data["contract_type"] == 0).astype(int) * 3  # Month-to-month contracts are higher risk
        + (data["payment_method"] == 0).astype(int) * 2  # Electronic check is higher risk
        - data["tech_support"] * 1  # Having tech support reduces risk
        - data["online_backup"] * 1  # Having online backup reduces risk
        - data["online_security"] * 1  # Having online security reduces risk
    )

    # Convert risk factors to binary churn (threshold chosen to get roughly balanced classes)
    data["churn"] = (risk_factors > 5).astype(int)
    data["churn_probability"] = np.round(np.clip(risk_factors / 10, 0, 1), 2)  # Probability of churn

    return pd.DataFrame(data)


def generate_sentiment_data(n_samples=20, random_seed=42):
    """Generate synthetic sentiment analysis data for text classification testing.

    The data follows the structure:
    - text: str (review text)
    - sentiment: str (positive, negative, or neutral)
    """
    np.random.seed(random_seed)

    positive_texts = [
        "This product exceeded my expectations!",
        "Great service and amazing quality",
        "I absolutely love this product",
        "Best purchase I've ever made",
        "Fantastic experience overall",
        "The customer service was exceptional",
        "Very satisfied with my purchase",
        "Would definitely recommend to friends",
        "Works perfectly for what I need",
        "Very happy with the quality and speed of delivery",
    ]

    negative_texts = [
        "Very disappointed with the quality",
        "Would not recommend this",
        "Poor service and slow delivery",
        "Product broke after first use",
        "Waste of money",
        "Customer service was terrible",
        "Not as described in the listing",
        "Overpriced for what you get",
        "Save your money and buy something else",
        "Regret making this purchase",
    ]

    neutral_texts = [
        "Product was ok, nothing special",
        "Meets basic expectations",
        "Delivery was on time",
        "Average quality for the price",
        "Some features work well, others don't",
        "Neither impressed nor disappointed",
        "Does the job but could be better",
        "Might purchase again, still deciding",
        "Mixed feelings about this product",
        "Not bad, not great",
    ]

    # Create balanced dataset with roughly equal numbers of each sentiment
    data = []
    categories = ["positive", "negative", "neutral"]
    text_sources = [positive_texts, negative_texts, neutral_texts]

    for i in range(n_samples):
        cat_idx = i % 3
        text_idx = np.random.randint(0, len(text_sources[cat_idx]))
        data.append({"text": text_sources[cat_idx][text_idx], "sentiment": categories[cat_idx]})

    # Shuffle the data
    np.random.shuffle(data)

    return pd.DataFrame(data)


def generate_product_recommendation_data(n_samples=30, random_seed=42):
    """Generate synthetic product recommendation data.

    The data follows the structure:
    - order_id: str (unique order identifier)
    - style: str (product style code)
    - category: str (product category)
    - customer_id: str (customer identifier)
    """
    np.random.seed(random_seed)

    # Define product categories and styles
    categories = ["Clothing", "Electronics", "Home", "Beauty", "Books"]

    # Generate unique style codes for each category
    styles_by_category = {}
    for category in categories:
        prefix = category[:3].upper()
        styles_by_category[category] = [f"{prefix}{100 + i}" for i in range(10)]

    # Flatten all styles
    all_styles = [style for styles in styles_by_category.values() for style in styles]

    # Create customer IDs
    customer_ids = [f"CUST{1000 + i}" for i in range(10)]

    # Generate purchase patterns
    # Each customer has preferences for certain categories and styles
    customer_preferences = {}
    for cust_id in customer_ids:
        # Pick 2-3 preferred categories
        preferred_cats = np.random.choice(categories, size=np.random.randint(2, 4), replace=False)
        # For each preferred category, pick 2-3 preferred styles
        preferred_styles = []
        for cat in preferred_cats:
            preferred_styles.extend(
                np.random.choice(styles_by_category[cat], size=np.random.randint(2, 4), replace=False)
            )
        customer_preferences[cust_id] = preferred_styles

    # Generate orders
    data = []
    order_ids = []

    # Create order IDs
    for i in range(n_samples // 3):  # Each order will have multiple items
        order_ids.append(f"ORD{10000 + i}")

    # Generate order data
    for order_id in order_ids:
        # Pick a random customer
        customer_id = np.random.choice(customer_ids)
        # Decide number of items in this order (2-5)
        n_items = np.random.randint(2, 6)

        # 70% chance the customer buys from their preferences
        if np.random.random() < 0.7:
            # Pick from their preferred styles
            styles = np.random.choice(
                customer_preferences[customer_id],
                size=min(n_items, len(customer_preferences[customer_id])),
                replace=False,
            )
        else:
            # Pick random styles
            styles = np.random.choice(all_styles, size=n_items, replace=False)

        # Add items to data
        for style in styles:
            # Find category of this style
            for cat, cat_styles in styles_by_category.items():
                if style in cat_styles:
                    category = cat
                    break

            data.append({"order_id": order_id, "style": style, "category": category, "customer_id": customer_id})

    return pd.DataFrame(data)


def generate_time_series_data(n_samples=60, random_seed=42):
    """Generate synthetic time series data for forecasting testing.

    The data follows the structure:
    - date: date (time index)
    - sales: float (sales amount)
    - promo: int (0=no promotion, 1=promotion running)
    - holiday: int (0=no holiday, 1=holiday)
    - day_of_week: int (0-6, Monday=0)
    """
    np.random.seed(random_seed)

    # Generate dates
    start_date = pd.to_datetime("2023-01-01")
    dates = [start_date + pd.Timedelta(days=i) for i in range(n_samples)]

    # Generate features
    day_of_week = [date.weekday() for date in dates]

    # Holidays (random 7% of days are holidays)
    holiday = np.zeros(n_samples)
    holiday_indices = np.random.choice(range(n_samples), size=int(n_samples * 0.07), replace=False)
    holiday[holiday_indices] = 1

    # Promotions (random 20% of days have promotions)
    promo = np.zeros(n_samples)
    promo_indices = np.random.choice(range(n_samples), size=int(n_samples * 0.2), replace=False)
    promo[promo_indices] = 1

    # Generate sales with trend, seasonality, and effects of promotions and holidays
    # Base trend (slightly increasing)
    trend = np.linspace(0, 20, n_samples)

    # Weekly seasonality (higher on weekends)
    seasonality = np.array([5 if dow >= 5 else 0 for dow in day_of_week])

    # Promotion and holiday effects
    promo_effect = promo * 25
    holiday_effect = holiday * 15

    # Combine effects
    sales = 100 + trend + seasonality + promo_effect + holiday_effect

    # Add noise
    sales += np.random.normal(0, 10, n_samples)

    # Ensure sales are positive
    sales = np.maximum(0, sales)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "date": dates,
            "sales": np.round(sales, 2),
            "promo": promo.astype(int),
            "holiday": holiday.astype(int),
            "day_of_week": day_of_week,
        }
    )

    return data


def verify_prediction(prediction, expected_schema=None):
    """Verify that a prediction matches expected format."""
    assert isinstance(prediction, dict), "Prediction should be a dictionary"
    assert len(prediction) > 0, "Prediction should not be empty"

    if expected_schema:
        schema_keys = getattr(expected_schema, "model_fields", None)
        if schema_keys is not None:
            schema_keys = set(schema_keys.keys())
        else:
            schema_keys = set(expected_schema.keys())

        assert (
            set(prediction.keys()) == schema_keys
        ), f"Prediction keys {prediction.keys()} don't match schema keys {schema_keys}"

    # Check first value to ensure it's of expected type
    output_value = list(prediction.values())[0]
    if isinstance(output_value, list):
        # If the output is a list, check that it's not empty
        assert len(output_value) > 0, "Prediction list should not be empty"
    else:
        # If output is not a list, check that it's one of the expected types
        assert isinstance(
            output_value, (int, float, str)
        ), f"Prediction value should be numeric, string, or list, got {type(output_value)}"


def verify_model_description(description):
    """Verify that a model description contains expected fields."""
    assert isinstance(description, ModelDescription), "Model description should be a 'ModelDescription' object"
    required_fields = ["intent", "schemas", "code"]
    for field in required_fields:
        assert hasattr(description, field), f"Model description missing required field: {field}"


def cleanup_files(model_dir=None):
    """Clean up any files created during tests."""
    files_to_clean = [
        "plexe.log",
        "*.pmb",
        "*.tar.gz",
    ]

    # Clean up files in current directory
    for pattern in files_to_clean:
        try:
            for file in Path(".").glob(pattern):
                if file.is_file():
                    file.unlink()
        except Exception as e:
            print(f"Failed to clean up {pattern}: {e}")

    # Clean up files in model directory
    if model_dir is not None and Path(model_dir).exists():
        try:
            # Use rmtree to recursively remove directory and contents
            shutil.rmtree(model_dir, ignore_errors=True)
        except Exception as e:
            print(f"Failed to clean up {model_dir}: {e}")
            # If rmtree fails, try to at least clean individual files
            for file in Path(model_dir).glob("*"):
                try:
                    if file.is_file():
                        file.unlink(missing_ok=True)
                    elif file.is_dir():
                        shutil.rmtree(file, ignore_errors=True)
                except Exception as e:
                    print(f"Failed to clean up {file}: {e}")
