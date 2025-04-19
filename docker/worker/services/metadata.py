import os
from datetime import datetime
from pymongo import MongoClient


class MetadataService:
    """Service for interacting with model metadata"""

    # Connect to MongoDB
    client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
    db = client[os.getenv("MONGODB_DB", "plexe")]

    @classmethod
    def _serialize_metrics(cls, metrics):
        """Convert metrics object to a serializable format"""
        if not metrics:
            return None

        # If it's already a dict, return it
        if isinstance(metrics, dict):
            return metrics

        # Handle Metric objects from plexe
        serialized = {}
        if hasattr(metrics, "name"):
            serialized["name"] = metrics.name
        if hasattr(metrics, "value"):
            serialized["value"] = metrics.value

        # Handle MetricComparator object
        if hasattr(metrics, "comparator"):
            comparator = {}
            if hasattr(metrics.comparator, "comparison_method"):
                # Convert Enum to string
                if hasattr(metrics.comparator.comparison_method, "value"):
                    comparator["comparison_method"] = metrics.comparator.comparison_method.value
                else:
                    comparator["comparison_method"] = str(metrics.comparator.comparison_method)
            if hasattr(metrics.comparator, "target"):
                comparator["target"] = metrics.comparator.target
            serialized["comparator"] = comparator

        return serialized

    @classmethod
    def update_model_status(cls, model_id, status, metrics=None, error=None):
        """Update the status of a model"""
        update_data = {"status": status, "updated_at": datetime.utcnow()}

        if metrics:
            # Serialize the metrics for MongoDB
            update_data["metrics"] = cls._serialize_metrics(metrics)

        if error:
            update_data["error"] = str(error)

        cls.db.models.update_one({"model_id": model_id}, {"$set": update_data})

    @classmethod
    def update_job_status(cls, job_id, status, error=None):
        """Update the status of a job"""
        update_data = {"status": status, "updated_at": datetime.utcnow()}

        if error:
            update_data["error"] = str(error)

        cls.db.jobs.update_one({"job_id": job_id}, {"$set": update_data})
