"""Abstract transport interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Transport(Protocol):
    """Interface for LiDAR communication transports."""

    def open(self) -> int:
        """Open the connection. Returns 0 on success, -1 on failure."""
        ...

    def close(self) -> None:
        """Close the connection."""
        ...

    def read(self, max_bytes: int = 8192) -> bytes:
        """Read up to max_bytes. Returns empty bytes if nothing available."""
        ...

    def write(self, data: bytes) -> int:
        """Write data. Returns number of bytes written."""
        ...

    @property
    def is_open(self) -> bool:
        """Whether the transport is currently open."""
        ...
