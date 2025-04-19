import streamlit as st
import requests
import json
import time
import os

# Configure API URL
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="plexe UI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
st.sidebar.title("plexe")
st.sidebar.caption("✨ Natural Language ML Platform")
page = st.sidebar.radio("Navigation", ["Create Model", "Models", "Predictions"])


# Helper functions
def create_model(intent, input_schema, output_schema, example_data=None):
    """Create a model via the API"""
    payload = {
        "intent": intent,
        "input_schema": input_schema,
        "output_schema": output_schema,
    }

    if example_data is not None:
        payload["example_data"] = example_data

    response = requests.post(f"{API_URL}/models", json=payload)
    if response.status_code != 200:
        st.error(f"Error: {response.text}")
        return None
    return response.json()


def get_models():
    """Get all models from the API"""
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code != 200:
            st.error(f"Error: {response.text}")
            return None
        return response.json()
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None


def get_model(model_id):
    """Get a specific model's details"""
    response = requests.get(f"{API_URL}/models/{model_id}")
    if response.status_code != 200:
        st.error(f"Error: {response.text}")
        return None
    return response.json()


def get_job(job_id):
    """Get a job's status"""
    response = requests.get(f"{API_URL}/jobs/{job_id}")
    if response.status_code != 200:
        st.error(f"Error: {response.text}")
        return None
    return response.json()


def make_prediction(model_id, data):
    """Make a prediction with the model"""
    response = requests.post(f"{API_URL}/predictions/models/{model_id}", json={"data": data})
    if response.status_code != 200:
        st.error(f"Error: {response.text}")
        return None
    return response.json()


# Page: Create Model
if page == "Create Model":
    st.title("Create a New ML Model")

    # Example templates
    st.sidebar.header("Templates")
    template = st.sidebar.selectbox("Choose a template", ["Heart Attack Prediction", "Sentiment Analysis", "Custom"])

    if template == "Heart Attack Prediction":
        default_intent = "Predict whether a patient will have a heart attack based on clinical data."
        default_input_schema = {
            "age": "int",
            "cholesterol": "float",
            "exercise": "bool",
            "smoking": "bool",
            "blood_pressure": "float",
        }
        default_output_schema = {"heart_attack": "bool"}
        default_examples = [
            {
                "age": 45,
                "cholesterol": 230.0,
                "exercise": True,
                "smoking": False,
                "blood_pressure": 120.0,
                "heart_attack": False,
            },
            {
                "age": 62,
                "cholesterol": 300.0,
                "exercise": False,
                "smoking": True,
                "blood_pressure": 155.0,
                "heart_attack": True,
            },
            {
                "age": 52,
                "cholesterol": 210.0,
                "exercise": True,
                "smoking": True,
                "blood_pressure": 140.0,
                "heart_attack": False,
            },
        ]
    elif template == "Sentiment Analysis":
        default_intent = "Classify the sentiment of customer reviews as positive, negative, or neutral."
        default_input_schema = {"review_text": "str"}
        default_output_schema = {"sentiment": "str"}
        default_examples = [
            {"review_text": "This product is amazing, I love it!", "sentiment": "positive"},
            {"review_text": "Decent quality but overpriced.", "sentiment": "neutral"},
            {"review_text": "Terrible experience, would not recommend.", "sentiment": "negative"},
        ]
    else:  # Custom
        default_intent = "Describe what your model should do..."
        default_input_schema = {"feature1": "str", "feature2": "int"}
        default_output_schema = {"prediction": "str"}
        default_examples = [
            {"feature1": "example1", "feature2": 1, "prediction": "result1"},
            {"feature1": "example2", "feature2": 2, "prediction": "result2"},
        ]

    # Model definition form
    with st.form("model_form"):
        st.header("Model Definition")

        intent = st.text_area("Intent (describe what the model should do)", default_intent)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Schema")
            input_schema_str = st.text_area("Input Schema (JSON)", json.dumps(default_input_schema, indent=2))

        with col2:
            st.subheader("Output Schema")
            output_schema_str = st.text_area("Output Schema (JSON)", json.dumps(default_output_schema, indent=2))

        include_examples = st.checkbox("Include training examples", True)

        if include_examples:
            example_data_str = st.text_area("Example Data (JSON array)", json.dumps(default_examples, indent=2))

        submitted = st.form_submit_button("Create Model")

        if submitted:
            try:
                input_schema = json.loads(input_schema_str)
                output_schema = json.loads(output_schema_str)

                if include_examples:
                    example_data = json.loads(example_data_str)
                else:
                    example_data = None

                with st.spinner("Creating model..."):
                    result = create_model(intent, input_schema, output_schema, example_data)

                if result:
                    st.success(f"Model created! Model ID: {result['model_id']}")
                    st.info(f"Job ID: {result['job_id']}")

                    st.session_state["last_job_id"] = result["job_id"]
                    st.session_state["last_model_id"] = result["model_id"]

                    # Poll job status
                    if st.checkbox("Monitor job status"):
                        placeholder = st.empty()
                        job_complete = False

                        while not job_complete:
                            job_status = get_job(result["job_id"])
                            if job_status:
                                placeholder.json(job_status)
                                if job_status["status"] in ["completed", "failed"]:
                                    job_complete = True
                            time.sleep(3)

                        if job_status["status"] == "completed":
                            st.success("Job completed successfully!")
                        else:
                            st.error(f"Job failed: {job_status.get('error', 'Unknown error')}")

            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON format: {str(e)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Page: Models
elif page == "Models":
    st.title("Available Models")

    if st.button("Refresh Models"):
        st.session_state["models"] = get_models()

    if "models" not in st.session_state:
        with st.spinner("Loading models..."):
            st.session_state["models"] = get_models()

    if st.session_state["models"]:
        for model in st.session_state["models"]:
            with st.expander(f"{model['intent']} (ID: {model['model_id']})"):
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Model Details")
                    st.write(f"Status: {model['status']}")
                    st.write(f"Created: {model['created_at']}")
                    if "updated_at" in model and model["updated_at"]:
                        st.write(f"Updated: {model['updated_at']}")

                with col2:
                    st.subheader("Schemas")
                    st.json(
                        {"input_schema": model.get("input_schema", {}), "output_schema": model.get("output_schema", {})}
                    )

                if model.get("metrics"):
                    st.subheader("Performance Metrics")
                    st.json(model["metrics"])

                if model["status"] == "ready":
                    if st.button("Use this model", key=f"use_{model['model_id']}"):
                        st.session_state["prediction_model"] = model
                        st.info("Switched to Predictions tab to use this model")
    else:
        st.info("No models found. Create one in the 'Create Model' tab.")

# Page: Predictions
elif page == "Predictions":
    st.title("Make Predictions")

    # Select model
    if "prediction_model" not in st.session_state and "models" in st.session_state:
        ready_models = [m for m in st.session_state["models"] if m["status"] == "ready"]
        if ready_models:
            st.session_state["prediction_model"] = ready_models[0]

    if "prediction_model" in st.session_state:
        model = st.session_state["prediction_model"]

        st.subheader(f"Model: {model['intent']}")
        st.write(f"ID: {model['model_id']}")

        # Input form based on model's input schema
        if model.get("input_schema"):
            with st.form("prediction_form"):
                st.subheader("Input Data")

                input_values = {}
                for field_name, field_type in model["input_schema"].items():
                    if field_type in ["str", "string"]:
                        input_values[field_name] = st.text_input(f"{field_name} (text)")
                    elif field_type in ["int", "integer"]:
                        input_values[field_name] = st.number_input(f"{field_name} (number)", step=1)
                    elif field_type in ["float", "number"]:
                        input_values[field_name] = st.number_input(f"{field_name} (decimal)")
                    elif field_type in ["bool", "boolean"]:
                        input_values[field_name] = st.checkbox(f"{field_name} (yes/no)")

                predict_button = st.form_submit_button("Make Prediction")

                if predict_button:
                    with st.spinner("Making prediction..."):
                        result = make_prediction(model["model_id"], input_values)

                    if result:
                        st.success("Prediction complete!")
                        st.subheader("Result")
                        st.json(result["prediction"])
        else:
            st.warning("Model schema information is not available")
    else:
        st.info("No ready models available. Create and train a model first.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("plexe © 2025")
