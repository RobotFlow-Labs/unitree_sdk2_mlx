"""UDP transport — works identically on macOS and Linux."""

from __future__ import annotations

import logging
import socket
import select

logger = logging.getLogger(__name__)


class UDPTransport:
    """UDP socket transport for LiDAR communication."""

    def __init__(
        self,
        lidar_port: int = 6101,
        lidar_ip: str = "192.168.1.62",
        local_port: int = 6201,
        local_ip: str = "192.168.1.2",
        recv_timeout: float = 0.01,
    ):
        self._lidar_addr = (lidar_ip, lidar_port)
        self._local_addr = (local_ip, local_port)
        self._timeout = recv_timeout
        self._sock: socket.socket | None = None

    def open(self) -> int:
        """Create and bind UDP socket. Returns 0 on success, -1 on failure."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind(self._local_addr)
            self._sock.setblocking(False)
            return 0
        except OSError as e:
            logger.error("Failed to open UDP socket on %s: %s", self._local_addr, e)
            self._sock = None
            return -1

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def read(self, max_bytes: int = 8192) -> bytes:
        if self._sock is None:
            return b""
        ready, _, _ = select.select([self._sock], [], [], self._timeout)
        if not ready:
            return b""
        try:
            data, _ = self._sock.recvfrom(max_bytes)
            return data
        except (OSError, BlockingIOError):
            return b""

    def write(self, data: bytes) -> int:
        if self._sock is None:
            return 0
        try:
            return self._sock.sendto(data, self._lidar_addr)
        except OSError:
            return 0

    @property
    def is_open(self) -> bool:
        return self._sock is not None
