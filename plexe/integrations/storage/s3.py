"""
Amazon S3 storage helper.

Provides reusable S3 operations as building blocks for ``WorkflowIntegration``
implementations. Custom integrations can compose with this helper to avoid
rewriting S3 upload/download logic.
"""

import logging
import os
import tarfile
import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from plexe.integrations.storage import StorageHelper
from plexe.constants import DirNames

logger = logging.getLogger(__name__)


class S3Helper(StorageHelper):
    """
    Amazon S3 storage helper.

    Usage::

        s3 = S3Helper(bucket="my-bucket", prefix="plexe", user_id="user123")
        s3.upload_file(local_path, s3.build_key(experiment_id, "model", "model.tar.gz"))
    """

    def __init__(self, bucket: str, prefix: str = "", user_id: str | None = None):
        self.bucket = bucket
        self.prefix = prefix
        self.user_id = user_id
        self.client = boto3.client("s3")

    @staticmethod
    def parse_uri(uri: str) -> tuple[str, str]:
        """
        Parse s3://bucket/prefix into (bucket, prefix).

        Args:
            uri: S3 URI (e.g., s3://my-bucket/some/prefix)

        Returns:
            (bucket, prefix) tuple
        """
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI (expected s3://...): {uri}")
        parts = uri[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
        return bucket, prefix

    def build_key(self, experiment_id: str, *parts: str) -> str:
        """
        Build S3 key with user/experiment scoping.

        Args:
            experiment_id: Experiment identifier
            *parts: Additional path components

        Returns:
            Full S3 key (e.g., prefix/users/uid/experiments/eid/parts...)
        """
        if self.user_id:
            base_parts = [self.prefix, "users", self.user_id, "experiments", experiment_id]
        else:
            base_parts = [self.prefix, experiment_id]
        key_parts = [p for p in base_parts + list(parts) if p]
        return "/".join(key_parts)

    def upload_file(self, local_path: Path, key: str) -> str:
        """
        Upload file to S3.

        Returns:
            Full S3 URI (s3://bucket/key)
        """
        self.client.upload_file(str(local_path), self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def download_file(self, key: str, local_path: Path) -> None:
        """Download file from S3."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(local_path))

    def object_exists(self, key: str) -> bool:
        """
        Check if S3 object exists.

        Raises:
            ClientError: If S3 access fails for reasons other than "not found"
        """
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NotFound", "NoSuchKey"):
                return False
            logger.error(f"S3 access error for s3://{self.bucket}/{key}: {e}")
            raise

    def download_directory(self, uri: str, local_dir: Path) -> str:
        """
        Download a Spark parquet directory (multiple part files) from S3.

        Handles S3 pagination for directories with >1000 files.

        Args:
            uri: S3 URI to parquet directory
            local_dir: Base local directory (subdirectory created from URI)

        Returns:
            Local directory path containing the downloaded files
        """
        bucket, prefix = self.parse_uri(uri)
        if not prefix:
            raise ValueError(f"Invalid S3 URI (expected s3://bucket/key): {uri}")

        logger.info(f"Downloading directory from S3: {uri}")
        paginator = self.client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        target_dir = local_dir / Path(prefix).name
        target_dir.mkdir(parents=True, exist_ok=True)

        file_count = 0
        for page in pages:
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                obj_key = obj["Key"]
                relative_path = obj_key[len(prefix) :].lstrip("/")
                if not relative_path:
                    continue
                local_file = target_dir / relative_path
                local_file.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(bucket, obj_key, str(local_file))
                file_count += 1

        if file_count == 0:
            raise ValueError(f"No objects found at {uri}")

        logger.info(f"Downloaded {file_count} files to {target_dir}")
        return str(target_dir)

    def tar_and_upload(self, local_dir: Path, s3_key: str) -> str:
        """
        Create tarball from directory and upload to S3.

        Returns:
            Full S3 URI (s3://bucket/key)
        """
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tarball_path = tmp.name
        try:
            logger.info(f"Creating tarball from {local_dir}...")
            with tarfile.open(tarball_path, "w:gz") as tar:
                tar.add(local_dir, arcname=DirNames.BUILD_DIR)
            tarball_size_mb = Path(tarball_path).stat().st_size / (1024**2)
            logger.info(f"Tarball size: {tarball_size_mb:.1f} MB")
            return self.upload_file(Path(tarball_path), s3_key)
        finally:
            if Path(tarball_path).exists():
                os.remove(tarball_path)

    def download_and_extract_tar(self, s3_key: str, extract_to: Path) -> None:
        """
        Download tarball from S3 and extract.

        Raises:
            ValueError: If tarball contains paths outside extract_to (path traversal)
        """
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tarball_path = tmp.name
        try:
            self.download_file(s3_key, Path(tarball_path))
            tarball_size_mb = Path(tarball_path).stat().st_size / (1024**2)
            logger.info(f"Downloaded tarball: {tarball_size_mb:.1f} MB")
            logger.info(f"Extracting tarball to {extract_to}...")
            with tarfile.open(tarball_path, "r:gz") as tar:
                extract_to_resolved = extract_to.resolve()
                safe_members = []
                for member in tar.getmembers():
                    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                        logger.warning(f"Skipping unsafe tar member: {member.name}")
                        continue
                    member_path = (extract_to / member.name).resolve()
                    if not member_path.is_relative_to(extract_to_resolved):
                        raise ValueError(f"Tarball contains unsafe path traversal: {member.name}")
                    safe_members.append(member)
                tar.extractall(path=extract_to, members=safe_members)
        finally:
            if Path(tarball_path).exists():
                os.remove(tarball_path)

    def handle_download_error(self, error: ClientError, reference: str, context: str = "") -> None:
        """Raise appropriate exception for S3 download errors."""
        error_code = error.response["Error"]["Code"]
        if error_code in ("404", "NoSuchKey"):
            raise FileNotFoundError(f"Not found{context}: {reference}") from error
        elif error_code in ("403", "AccessDenied"):
            raise PermissionError(f"Access denied{context}: {reference}") from error
        else:
            raise
