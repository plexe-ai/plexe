"""
S3 utilities for downloading datasets.

Provides simple helpers for downloading S3 URIs to local disk.
"""

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def download_s3_uri(s3_uri: str, local_dir: Path | None = None) -> str:
    """
    Download S3 URI to local directory.

    Handles both:
    - Single files: s3://bucket/path/file.parquet
    - Directories (multi-part parquet): s3://bucket/path/dir/

    Args:
        s3_uri: S3 URI to download
        local_dir: Local directory to download to (default: temp directory)

    Returns:
        Local path to downloaded file/directory
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    logger.info(f"Downloading from S3: {s3_uri}")

    # Parse S3 URI
    s3_path = s3_uri[5:]  # Remove 's3://'
    parts = s3_path.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    # Create local directory
    if local_dir is None:
        local_dir = Path(tempfile.mkdtemp(prefix="s3_download_"))
    else:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

    # Initialize S3 client (boto3 imported lazily — optional dependency)
    import boto3

    s3_client = boto3.client("s3")

    # Check if it's a directory or single file
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=10)

        if "Contents" not in response or len(response["Contents"]) == 0:
            raise ValueError(f"No objects found at S3 URI: {s3_uri}")

        # If multiple files or key ends with '/', treat as directory
        if len(response["Contents"]) > 1 or key.endswith("/"):
            # Directory: download all files
            logger.info("Downloading directory from S3 (multi-part parquet)")

            # List all objects with this prefix
            paginator = s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=key)

            downloaded_count = 0
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        obj_key = obj["Key"]
                        # Skip directories (keys ending with /)
                        if obj_key.endswith("/"):
                            continue

                        # Preserve directory structure relative to prefix
                        relative_path = obj_key[len(key) :].lstrip("/")
                        local_file = local_dir / relative_path

                        # Create parent directories
                        local_file.parent.mkdir(parents=True, exist_ok=True)

                        # Download file
                        s3_client.download_file(bucket, obj_key, str(local_file))
                        downloaded_count += 1
                        logger.debug(f"Downloaded: {relative_path}")

            logger.info(f"✓ Downloaded {downloaded_count} files to {local_dir}")
            return str(local_dir)

        else:
            # Single file: download directly
            filename = Path(key).name
            local_file = local_dir / filename

            s3_client.download_file(bucket, key, str(local_file))
            logger.info(f"✓ Downloaded file to {local_file}")
            return str(local_file)

    except Exception as e:
        logger.error(f"Failed to download from S3: {e}")
        raise
