"""
Storage helper interface and implementations.

Provides reusable building blocks for ``WorkflowIntegration`` implementations
that need to interact with cloud or remote storage backends.

Shipped implementations:
- ``S3Helper`` — Amazon S3
- ``AzureBlobHelper`` — Azure Blob Storage (stub, not yet implemented)
- ``GCSHelper`` — Google Cloud Storage (stub, not yet implemented)
"""

from abc import ABC, abstractmethod
from pathlib import Path


class StorageHelper(ABC):
    """
    Abstract base for cloud/remote storage helpers.

    Defines the common operations that any storage backend must support.
    Implementations are used as composable building blocks inside
    ``WorkflowIntegration`` subclasses — they are not integration classes
    themselves.
    """

    @abstractmethod
    def upload_file(self, local_path: Path, key: str) -> str:
        """
        Upload a local file to remote storage.

        Args:
            local_path: Local file to upload
            key: Remote storage key/path

        Returns:
            Full URI of the uploaded file
        """

    @abstractmethod
    def download_file(self, key: str, local_path: Path) -> None:
        """
        Download a remote file to local filesystem.

        Args:
            key: Remote storage key/path
            local_path: Local destination path
        """

    @abstractmethod
    def object_exists(self, key: str) -> bool:
        """
        Check whether an object exists in remote storage.

        Args:
            key: Remote storage key/path

        Returns:
            True if the object exists
        """

    @abstractmethod
    def download_directory(self, uri: str, local_dir: Path) -> str:
        """
        Download a remote directory (e.g., Spark parquet output) to local filesystem.

        Args:
            uri: Full URI of the remote directory
            local_dir: Base local directory (a subdirectory will be created)

        Returns:
            Local directory path containing the downloaded files
        """
