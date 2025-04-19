import os
import json
import redis


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
    def dequeue_job(cls, timeout=30):
        """Get a job from the queue"""
        result = cls.redis_client.brpop("plexe:jobs", timeout)

        if result:
            job_data = json.loads(result[1])
            return job_data

        return None
