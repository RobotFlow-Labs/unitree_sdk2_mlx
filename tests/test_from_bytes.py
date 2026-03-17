"""Tests for from_bytes deserialization — verifies struct alignment against C++ protocol.

These tests build raw binary payloads matching the C++ struct layout and verify
that Python deserialization produces correct values. This catches struct
misalignment bugs that constructor-based tests miss.
"""

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
    SIZE_DATA_INFO,
    SIZE_INSIDE_STATE,
    SIZE_CALIB_PARAM,
    FMT_FRAME_HEADER,
    FMT_FRAME_TAIL,
    FMT_DATA_INFO,
    FMT_INSIDE_STATE,
    FMT_CALIB_PARAM,
    PACKET_POINT_3D,
    PACKET_POINT_2D,
    PACKET_IMU,
    MAX_POINTS_3D,
    MAX_POINTS_2D,
    PointData3D,
    PointData2D,
    ImuDataPacket,
    VersionDataPacket,
)
from unitree_sdk2_mlx.parser import FrameParser


def _build_point3d_payload(
    point_num: int = 50,
    range_val: int = 2000,
    intensity_val: int = 150,
) -> bytes:
    """Build a raw 3D point data payload matching C++ LidarPointData struct."""
    parts = []

    # DataInfo: seq=42, payload_size=0, stamp_sec=1000, stamp_nsec=500000000
    parts.append(struct.pack(FMT_DATA_INFO, 42, 0, 1000, 500000000))

    # InsideState: 2 uint32 + 7 floats
    parts.append(struct.pack(FMT_INSIDE_STATE,
                             100, 200,           # sys/com rotation period
                             5.2,                 # dirty_index
                             0.01, 0.02,          # packet_lost_up/down
                             25.0, 3.3, 1.2,      # apd_temp, apd_voltage, laser_voltage
                             26.5))               # imu_temperature

    # CalibParam: 8 floats
    parts.append(struct.pack(FMT_CALIB_PARAM,
                             0.05,   # a_axis_dist
                             0.02,   # b_axis_dist
                             0.01,   # theta_angle_bias
                             0.005,  # alpha_angle_bias
                             0.1,    # beta_angle
                             0.2,    # xi_angle
                             0.0,    # range_bias
                             0.001)) # range_scale (mm to m)

    # 8 float scan fields + 1 uint32 point_num
    parts.append(struct.pack("<8fI",
                             0.5,           # com_horizontal_angle_start
                             0.01,          # com_horizontal_angle_step
                             0.1,           # scan_period
                             0.1,           # range_min
                             50.0,          # range_max
                             -0.785,        # angle_min (~-pi/4)
                             0.02,          # angle_increment
                             0.001,         # time_increment
                             point_num))    # point_num

    # Ranges: 300 uint16
    ranges = np.full(MAX_POINTS_3D, range_val, dtype=np.uint16)
    # Set first few to 0 (invalid)
    ranges[0] = 0
    ranges[1] = 0
    parts.append(ranges.tobytes())

    # Intensities: 300 uint8
    intensities = np.full(MAX_POINTS_3D, intensity_val, dtype=np.uint8)
    parts.append(intensities.tobytes())

    return b"".join(parts)


def _build_point2d_payload(
    point_num: int = 100,
    range_val: int = 3000,
) -> bytes:
    """Build a raw 2D point data payload matching C++ Lidar2DPointData struct."""
    parts = []

    # DataInfo
    parts.append(struct.pack(FMT_DATA_INFO, 10, 0, 2000, 0))

    # InsideState
    parts.append(struct.pack(FMT_INSIDE_STATE, 100, 200, 3.0, 0.0, 0.0, 25.0, 3.3, 1.2, 26.5))

    # CalibParam
    parts.append(struct.pack(FMT_CALIB_PARAM, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001))

    # 6 floats + 1 uint32
    parts.append(struct.pack("<6fI",
                             0.1,    # scan_period
                             0.1,    # range_min
                             50.0,   # range_max
                             -1.57,  # angle_min
                             0.003,  # angle_increment
                             0.001,  # time_increment
                             point_num))

    # Ranges: 1800 uint16
    parts.append(np.full(MAX_POINTS_2D, range_val, dtype=np.uint16).tobytes())

    # Intensities: 1800 uint8
    parts.append(np.full(MAX_POINTS_2D, 200, dtype=np.uint8).tobytes())

    return b"".join(parts)


def _frame_packet(packet_type: int, payload: bytes) -> bytes:
    """Wrap payload in a valid frame with header, CRC, and tail."""
    total = SIZE_FRAME_HEADER + len(payload) + SIZE_FRAME_TAIL
    header = struct.pack(FMT_FRAME_HEADER, FRAME_HEADER_MAGIC, packet_type, total)
    crc_data = header + payload
    crc_val = crc32(crc_data)
    tail = struct.pack(FMT_FRAME_TAIL, crc_val, packet_type, b"\x00\x00", FRAME_TAIL_MAGIC)
    return header + payload + tail


# ─── 3D Point Data from_bytes tests ─────────────────────────────────────


class TestPointData3DFromBytes:
    def test_basic_deserialization(self):
        payload = _build_point3d_payload(point_num=50, range_val=2000)
        pkt = PointData3D.from_bytes(payload)

        assert pkt.info.seq == 42
        assert abs(pkt.info.stamp - 1000.5) < 1e-6
        assert pkt.point_num == 50
        assert abs(pkt.calib.a_axis_dist - 0.05) < 1e-6
        assert abs(pkt.calib.range_scale - 0.001) < 1e-6
        assert abs(pkt.com_horizontal_angle_start - 0.5) < 1e-6
        assert abs(pkt.scan_period - 0.1) < 1e-6
        assert abs(pkt.angle_min - (-0.785)) < 1e-3
        assert abs(pkt.dirty_index - 5.2) < 1e-5

    def test_ranges_and_intensities_alignment(self):
        """Verify ranges/intensities arrays are correctly aligned after header."""
        payload = _build_point3d_payload(point_num=50, range_val=2000, intensity_val=150)
        pkt = PointData3D.from_bytes(payload)

        # First two ranges should be 0 (set to invalid in builder)
        assert pkt.ranges[0] == 0
        assert pkt.ranges[1] == 0
        # Rest should be 2000
        assert pkt.ranges[2] == 2000
        assert pkt.ranges[299] == 2000
        # Intensities
        assert pkt.intensities[0] == 150
        assert pkt.intensities[299] == 150

    def test_full_pipeline_via_parser(self):
        """Feed a framed 3D packet through the parser and verify output."""
        payload = _build_point3d_payload(point_num=50, range_val=5000)
        framed = _frame_packet(PACKET_POINT_3D, payload)

        parser = FrameParser()
        parser.feed(framed)
        result = parser.parse_one()

        assert result is not None
        assert result.packet_type == PACKET_POINT_3D
        assert result.payload.point_num == 50
        assert result.payload.ranges[2] == 5000

    def test_transform_from_raw_bytes(self):
        """Full end-to-end: raw bytes -> parse -> transform -> point cloud."""
        from unitree_sdk2_mlx.transforms import transform_3d

        payload = _build_point3d_payload(point_num=10, range_val=3000)
        pkt = PointData3D.from_bytes(payload)
        pts = transform_3d(pkt)

        # First 2 points are invalid (ranges=0), so 8 valid points
        assert pts.shape == (8, 6)
        # All points should have finite coordinates
        assert np.all(np.isfinite(pts[:, :3]))
        # Intensities should be 150
        np.testing.assert_array_equal(pts[:, 3], 150.0)


# ─── 2D Point Data from_bytes tests ─────────────────────────────────────


class TestPointData2DFromBytes:
    def test_basic_deserialization(self):
        payload = _build_point2d_payload(point_num=100, range_val=3000)
        pkt = PointData2D.from_bytes(payload)

        assert pkt.info.seq == 10
        assert pkt.point_num == 100
        assert abs(pkt.calib.a_axis_dist - 0.05) < 1e-6
        assert pkt.ranges[0] == 3000
        assert pkt.intensities[0] == 200

    def test_full_pipeline_via_parser(self):
        payload = _build_point2d_payload(point_num=100)
        framed = _frame_packet(PACKET_POINT_2D, payload)

        parser = FrameParser()
        parser.feed(framed)
        result = parser.parse_one()

        assert result is not None
        assert result.packet_type == PACKET_POINT_2D
        assert result.payload.point_num == 100


# ─── Version data from_bytes test ───────────────────────────────────────


def test_version_from_bytes():
    from unitree_sdk2_mlx.protocol import FMT_VERSION_DATA
    payload = struct.pack(FMT_VERSION_DATA,
                          bytes([1, 1, 1, 1]),      # hw
                          bytes([2, 3, 3, 0]),      # sw
                          b"UniLidar L2\x00" + b"\x00" * 13,  # name (24 bytes)
                          b"20250304",              # date (8 bytes)
                          b"\x00" * 40)             # reserve

    ver = VersionDataPacket.from_bytes(payload)
    assert ver.hw_version == "1.1.1.1"
    assert ver.sw_version == "2.3.3.0"
    assert "UniLidar" in ver.name
