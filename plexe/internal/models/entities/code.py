"""
This module defines a Code dataclass for representing code objects passed around by agents.
"""

from dataclasses import dataclass, field


@dataclass
class Code:
    """Represents a code object."""

    code: str = field()
    performance: float = field(default=None)
