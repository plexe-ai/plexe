import os
from datetime import datetime
from pymongo import MongoClient
from uuid import uuid4


class MetadataService:
    """Service for interacting with model metadata"""

    # Connect to MongoDB
    client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
    db = client[os.getenv("MONGODB_DB", "plexe")]

    @classmethod
    def create_model_entry(cls, request):
        """Create a new model entry in the database"""
        model_id = str(uuid4())

        cls.db.models.insert_one(
            {
                "model_id": model_id,
                "status": "pending",
                "intent": request.intent,
                "input_schema": request.input_schema,
                "output_schema": request.output_schema,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
        )

        return model_id

    @classmethod
    def get_model(cls, model_id):
        """Get model metadata by ID"""
        model = cls.db.models.find_one({"model_id": model_id})
        if model:
            return model
        return None

    @classmethod
    def list_models(cls):
        """List all models"""
        return list(cls.db.models.find())

    @classmethod
    def update_model_status(cls, model_id, status, metrics=None, error=None):
        """Update the status of a model"""
        update_data = {"status": status, "updated_at": datetime.utcnow()}

        if metrics:
            # Handle metrics - convert to dict if needed
            if hasattr(metrics, "__dict__"):
                update_data["metrics"] = metrics.__dict__
            else:
                update_data["metrics"] = metrics

        if error:
            update_data["error"] = str(error)

        cls.db.models.update_one({"model_id": model_id}, {"$set": update_data})

    @classmethod
    def create_job_entry(cls, job_id, job_type, data):
        """Create a new job entry"""
        cls.db.jobs.insert_one(
            {
                "job_id": job_id,
                "type": job_type,
                "status": "pending",
                "data": data,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
        )

    @classmethod
    def get_job(cls, job_id):
        """Get job by ID"""
        return cls.db.jobs.find_one({"job_id": job_id})

    @classmethod
    def update_job_status(cls, job_id, status, error=None):
        """Update the status of a job"""
        update_data = {"status": status, "updated_at": datetime.utcnow()}

        if error:
            update_data["error"] = str(error)

        cls.db.jobs.update_one({"job_id": job_id}, {"$set": update_data})
