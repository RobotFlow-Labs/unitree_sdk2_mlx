"""CRC32 implementation matching unitree_lidar_utilities.h.

The C++ implementation uses polynomial 0xEDB88320 (reflected CRC-32),
which is identical to zlib.crc32().
"""

import zlib


def crc32(data: bytes) -> int:
    """Compute CRC32 matching the Unitree LiDAR protocol."""
    return zlib.crc32(data) & 0xFFFFFFFF
