import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
import plexe
import pandas as pd
import shutil

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("worker")

# Import services
from worker.services.queue import QueueService
from worker.services.metadata import MetadataService


# Ensure .smolcache directories exist
def ensure_cache_dirs():
    cache_dir = Path(".smolcache")
    models_dir = cache_dir / "models"
    extract_dir = cache_dir / "extracted"

    # Create directories with permissive permissions
    for directory in [cache_dir, models_dir, extract_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        # Ensure the directory has the correct permissions
        directory.chmod(0o777)

    logger.info(f"Cache directories created: {cache_dir}")
    return cache_dir


def convert_schema_types(schema_dict):
    """Convert string type representations to actual Python types"""
    if not schema_dict:
        return None

    type_mapping = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "string": str,
        "integer": int,
        "boolean": bool,
        "number": float,
    }

    converted_schema = {}
    for key, type_str in schema_dict.items():
        if isinstance(type_str, str) and type_str in type_mapping:
            converted_schema[key] = type_mapping[type_str]
        else:
            logger.error(f"Unsupported type: {type_str}")
            raise ValueError(f"Unsupported type: {type_str}")

    return converted_schema


def process_model_creation(data):
    """Process a model creation job"""
    model_id = data["model_id"]
    logger.info(f"Building model {model_id}")

    try:
        # Ensure cache directories exist
        cache_dir = ensure_cache_dirs()
        models_dir = cache_dir / "models"

        # Update model status
        MetadataService.update_model_status(model_id, "building")

        # Get schemas and convert string representations to actual Python types
        input_schema = convert_schema_types(data.get("input_schema"))
        output_schema = convert_schema_types(data.get("output_schema"))

        # Create model using plexe
        model = plexe.Model(intent=data["intent"], input_schema=input_schema, output_schema=output_schema)

        # Process datasets if available
        datasets = []
        if data.get("datasets"):
            for name, records in data["datasets"].items():
                df = pd.DataFrame(records)
                datasets.append(df)

        # If no datasets provided, generate synthetic data
        if not datasets and input_schema and output_schema:
            logger.info("No datasets provided, creating a synthetic dataset generator")
            dataset_gen = plexe.DatasetGenerator(
                description=data["intent"],
                provider=os.getenv("LLM_PROVIDER", "openai/gpt-4o-mini"),
                schema={**input_schema, **output_schema},
            )
            dataset_gen.generate(num_samples=50)  # Generate synthetic samples
            datasets.append(dataset_gen)

        # Build the model with datasets
        model.build(
            datasets=datasets,
            provider=os.getenv("LLM_PROVIDER", "openai/gpt-4o-mini"),
            timeout=data.get("timeout", 3600),
            max_iterations=data.get("max_iterations", 3),
        )

        # Save the model
        logger.info(f"Saving model {model_id}...")
        model_path = plexe.save_model(model, model_id)
        logger.info(f"Model saved to {model_path}")

        # Make sure model is also saved in the expected location for prediction service
        try:
            model_file = Path(model_path)
            target_path = models_dir / f"model-{model_id}.tar.gz"

            # Ensure we copy to the standard location
            if model_file.exists():
                logger.info(f"Copying model from {model_file} to {target_path}")
                shutil.copy2(model_file, target_path)
                logger.info(f"Model copied to {target_path}")

                # Also copy to root .smolcache for good measure
                root_path = cache_dir / f"{model_id}.tar.gz"
                shutil.copy2(model_file, root_path)
                logger.info(f"Model also copied to {root_path}")
        except Exception as e:
            logger.warning(f"Failed to copy model file: {e}")

        # Update metadata
        metrics = model.get_metrics()
        MetadataService.update_model_status(model_id=model_id, status="ready", metrics=metrics)

        logger.info(f"Model {model_id} built successfully")
        return True
    except Exception as e:
        logger.error(f"Error building model {model_id}: {str(e)}")
        MetadataService.update_model_status(model_id=model_id, status="error", error=str(e))
        return False


def main():
    """Main worker loop"""
    logger.info("Starting worker")

    # Ensure cache directories exist
    ensure_cache_dirs()

    while True:
        try:
            # Get job from queue
            job = QueueService.dequeue_job()

            if job:
                logger.info(f"Processing job {job['job_id']} of type {job['type']}")

                # Update job status
                MetadataService.update_job_status(job["job_id"], "processing")

                # Process job based on type
                if job["type"] == "MODEL_CREATE":
                    success = process_model_creation(job["data"])
                else:
                    logger.warning(f"Unknown job type: {job['type']}")
                    success = False

                # Update job status
                if success:
                    MetadataService.update_job_status(job["job_id"], "completed")
                else:
                    MetadataService.update_job_status(job["job_id"], "failed")
            else:
                # No jobs, sleep for a bit
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error processing job: {str(e)}")
            time.sleep(5)  # Sleep a bit longer on error


if __name__ == "__main__":
    main()
