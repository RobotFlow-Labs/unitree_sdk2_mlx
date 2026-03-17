"""Serial transport — macOS-adapted (/dev/cu.usbmodem*)."""

from __future__ import annotations

import glob
import sys
from typing import Optional

try:
    import serial

    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


def find_unitree_serial_port() -> Optional[str]:
    """Auto-detect the Unitree LiDAR serial port on macOS/Linux.

    Returns the first matching port or None.
    """
    if sys.platform == "darwin":
        # macOS: USB CDC-ACM devices appear as /dev/cu.usbmodem*
        candidates = sorted(glob.glob("/dev/cu.usbmodem*"))
    else:
        # Linux: /dev/ttyACM*
        candidates = sorted(glob.glob("/dev/ttyACM*"))

    return candidates[0] if candidates else None


class SerialTransport:
    """Serial port transport for LiDAR communication."""

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 4_000_000,
        timeout: float = 0.01,
    ):
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._serial: Optional[serial.Serial] = None

    def open(self) -> int:
        """Open serial port. Returns 0 on success, -1 on failure."""
        if not HAS_SERIAL:
            print("pyserial not installed. Run: uv pip install pyserial")
            return -1

        port = self._port or find_unitree_serial_port()
        if port is None:
            print("No Unitree LiDAR serial port found.")
            return -1

        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=self._baudrate,
                timeout=self._timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            return 0
        except (serial.SerialException, OSError) as e:
            print(f"Failed to open serial port {port}: {e}")
            self._serial = None
            return -1

    def close(self) -> None:
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
        self._serial = None

    def read(self, max_bytes: int = 8192) -> bytes:
        if self._serial is None or not self._serial.is_open:
            return b""
        try:
            waiting = self._serial.in_waiting
            if waiting == 0:
                return b""
            return self._serial.read(min(waiting, max_bytes))
        except (serial.SerialException, OSError):
            return b""

    def write(self, data: bytes) -> int:
        if self._serial is None or not self._serial.is_open:
            return 0
        try:
            return self._serial.write(data)
        except (serial.SerialException, OSError):
            return 0

    @property
    def is_open(self) -> bool:
        return self._serial is not None and self._serial.is_open
