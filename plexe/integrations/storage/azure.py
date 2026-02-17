"""
Azure Blob Storage helper (stub).

This module will provide reusable Azure Blob Storage operations for
``WorkflowIntegration`` implementations. It is not yet implemented.
"""

from pathlib import Path

from plexe.integrations.storage import StorageHelper


class AzureBlobHelper(StorageHelper):
    """
    Azure Blob Storage helper (not yet implemented).

    Will provide the same building-block operations as ``S3Helper``
    for Azure Blob Storage backends.
    """

    def upload_file(self, local_path: Path, key: str) -> str:
        raise NotImplementedError("Azure Blob Storage support is not yet implemented")

    def download_file(self, key: str, local_path: Path) -> None:
        raise NotImplementedError("Azure Blob Storage support is not yet implemented")

    def object_exists(self, key: str) -> bool:
        raise NotImplementedError("Azure Blob Storage support is not yet implemented")

    def download_directory(self, uri: str, local_dir: Path) -> str:
        raise NotImplementedError("Azure Blob Storage support is not yet implemented")
