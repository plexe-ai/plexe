"""
This module defines the "Artifact" dataclass, a simple representation of an external artifact.

An "external artifact" is a text or binary entity that is used by a model. The canonical example is
a blob containing the weights of a neural network. The Artifact class can be used either to point to
a file on disk by its path, or to hold the raw text or binary data in memory itself.
"""

import io
from typing import BinaryIO, Union
from pathlib import Path


class Artifact:
    """
    Represents a model artifact, which can either be a file path or raw text/binary data.

    When the artifact holds a path pointing to a file, the 'is_path' property is true and the path can
    be accessed via the 'path' attribute. When the artifact holds raw data, the 'is_data' property is true
    and the data can be accessed via the 'data' attribute. In either case, the 'type' property indicates
    whether the data (in memory or in the file) is text or binary.
    """

    def __init__(self, name: str, path: Path = None, handle: BinaryIO = None, data: bytes = None):
        self.name: str = name
        self.path: Path = path
        self.handle: BinaryIO = handle
        self.data: bytes = data

        if sum([path is not None, handle is not None, data is not None]) != 1:
            raise ValueError("Exactly one of 'handle', 'path', or 'data' must be provided.")

    def is_path(self) -> bool:
        """
        True if the artifact is a file path.
        """
        return self.path is not None

    def is_handle(self) -> bool:
        """
        True if the artifact is file path or file-like object.
        """
        return self.handle is not None

    def is_data(self) -> bool:
        """
        True if the artifact is a string or bytes object loaded in memory.
        """
        return self.data is not None

    def get_as_handle(self) -> BinaryIO:
        """
        Get the artifact as a file-like object.
        """
        if self.is_handle():
            return self.handle
        elif self.is_path():
            return open(self.path, "rb")
        elif self.is_data():
            return io.BytesIO(self.data)
        else:
            raise ValueError("Artifact does not have a valid handle, path, or data.")

    @staticmethod
    def from_path(path: Union[str, Path]):
        """
        Create an Artifact instance from a file path.
        """
        path = Path(path)
        return Artifact(name=path.name, path=path)

    @staticmethod
    def from_data(name: str, data: bytes):
        """
        Create an Artifact instance from an in-memory sequence of bytes.
        """
        return Artifact(name=name, data=data)
