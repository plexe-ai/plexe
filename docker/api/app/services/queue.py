import os
import json
import redis
from uuid import uuid4
from datetime import datetime

# Use relative import here
from .metadata import MetadataService


class QueueService:
    """Service for interacting with the job queue"""

    # Connect to Redis
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD", ""),
        db=int(os.getenv("REDIS_DB", 0)),
    )

    @classmethod
    def enqueue_job(cls, job_type, data):
        """Add a job to the queue"""
        job_id = str(uuid4())

        job_data = {"job_id": job_id, "type": job_type, "data": data, "created_at": datetime.utcnow().isoformat()}

        # Add job to Redis queue
        cls.redis_client.lpush("plexe:jobs", json.dumps(job_data))

        # Create job entry in metadata store
        MetadataService.create_job_entry(job_id, job_type, data)

        return job_id
