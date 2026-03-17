"""Shared test fixtures for building synthetic LiDAR packets."""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from unitree_sdk2_mlx.crc import crc32
from unitree_sdk2_mlx.protocol import (
    FRAME_HEADER_MAGIC,
    FRAME_TAIL_MAGIC,
    SIZE_FRAME_HEADER,
    SIZE_FRAME_TAIL,
    FMT_FRAME_HEADER,
    FMT_FRAME_TAIL,
    FMT_DATA_INFO,
    FMT_INSIDE_STATE,
    FMT_CALIB_PARAM,
    MAX_POINTS_3D,
    MAX_POINTS_2D,
    PACKET_POINT_3D,
    PACKET_POINT_2D,
    PACKET_IMU,
    PACKET_VERSION,
    FMT_IMU_DATA,
    FMT_VERSION_DATA,
)


def build_raw_3d_payload(
    seq: int = 1,
    point_num: int = 50,
    range_val: int = 2000,
    intensity_val: int = 128,
) -> bytes:
    """Build a raw binary 3D point payload matching C++ LidarPointData struct."""
    parts = []
    parts.append(struct.pack(FMT_DATA_INFO, seq, 0, 1000, 500000000))
    parts.append(struct.pack(FMT_INSIDE_STATE, 100, 200, 3.0, 0.0, 0.0, 25.0, 3.3, 1.2, 26.5))
    parts.append(struct.pack(FMT_CALIB_PARAM, 0.05, 0.02, 0.01, 0.005, 0.1, 0.2, 0.0, 0.001))
    parts.append(struct.pack("<8fI",
                             0.5, 0.01, 0.1, 0.1, 50.0,
                             -math.pi / 4, 0.02, 0.001,
                             point_num))
    parts.append(np.full(MAX_POINTS_3D, range_val, dtype=np.uint16).tobytes())
    parts.append(np.full(MAX_POINTS_3D, intensity_val, dtype=np.uint8).tobytes())
    return b"".join(parts)


def build_raw_imu_payload(seq: int = 1) -> bytes:
    """Build a raw IMU data payload."""
    return struct.pack(FMT_IMU_DATA,
                       seq, 56, 1000, 500000000,
                       0.0, 0.0, 0.0, 1.0,
                       0.1, 0.2, 0.3,
                       9.8, 0.0, 0.0)


def build_raw_version_payload() -> bytes:
    """Build a raw version data payload."""
    return struct.pack(FMT_VERSION_DATA,
                       bytes([1, 1, 1, 1]),
                       bytes([2, 3, 3, 0]),
                       b"UniLidar L2\x00" + b"\x00" * 13,
                       b"20250304",
                       b"\x00" * 40)


def frame_packet(packet_type: int, payload: bytes) -> bytes:
    """Wrap a payload in a valid framed packet with header, CRC, and tail."""
    total = SIZE_FRAME_HEADER + len(payload) + SIZE_FRAME_TAIL
    header = struct.pack(FMT_FRAME_HEADER, FRAME_HEADER_MAGIC, packet_type, total)
    crc_data = header + payload
    crc_val = crc32(crc_data)
    tail = struct.pack(FMT_FRAME_TAIL, crc_val, packet_type, b"\x00\x00", FRAME_TAIL_MAGIC)
    return header + payload + tail
