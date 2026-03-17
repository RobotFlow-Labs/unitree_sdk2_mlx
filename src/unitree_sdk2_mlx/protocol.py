"""Binary protocol structures matching unitree_lidar_protocol.h.

All structs are little-endian on the wire.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

FRAME_HEADER_MAGIC = bytes([0x55, 0xAA, 0x05, 0x0A])
FRAME_TAIL_MAGIC = bytes([0x00, 0xFF])

# Packet type IDs
PACKET_USER_CMD = 100
PACKET_ACK = 101
PACKET_POINT_3D = 102
PACKET_POINT_2D = 103
PACKET_IMU = 104
PACKET_VERSION = 105
PACKET_TIMESTAMP = 106
PACKET_WORK_MODE = 107
PACKET_IP_CONFIG = 108
PACKET_MAC_CONFIG = 109

# User command types
USER_CMD_RESET = 1
USER_CMD_STANDBY = 2
USER_CMD_VERSION_GET = 3
USER_CMD_LATENCY = 4
USER_CMD_CONFIG_RESET = 5
USER_CMD_CONFIG_GET = 6
USER_CMD_CONFIG_AUTO_STANDBY = 7

# ACK status
ACK_SUCCESS = 1
ACK_CRC_ERROR = 2
ACK_HEADER_ERROR = 3
ACK_BLOCK_ERROR = 4
ACK_WAIT_ERROR = 5

# ─── Struct formats (little-endian) ──────────────────────────────────────────

# FrameHeader: 4B magic + uint32 type + uint32 size = 12 bytes
FMT_FRAME_HEADER = "<4sII"
SIZE_FRAME_HEADER = struct.calcsize(FMT_FRAME_HEADER)  # 12

# FrameTail: uint32 crc + uint32 msg_type_check + 2B reserve + 2B tail = 12 bytes
FMT_FRAME_TAIL = "<II2s2s"
SIZE_FRAME_TAIL = struct.calcsize(FMT_FRAME_TAIL)  # 12

# TimeStamp: uint32 sec + uint32 nsec = 8 bytes
FMT_TIMESTAMP = "<II"
SIZE_TIMESTAMP = 8

# DataInfo: uint32 seq + uint32 payload_size + TimeStamp = 16 bytes
FMT_DATA_INFO = "<IIII"
SIZE_DATA_INFO = 16

# LidarCalibParam: 8 floats = 32 bytes
FMT_CALIB_PARAM = "<8f"
SIZE_CALIB_PARAM = 32

# LidarInsideState: 2 uint32 + 7 floats = 36 bytes
FMT_INSIDE_STATE = "<II7f"
SIZE_INSIDE_STATE = 36

# LidarImuData: DataInfo(16) + 4 floats quat + 3 floats gyro + 3 floats accel = 56 bytes
FMT_IMU_DATA = "<IIII4f3f3f"
SIZE_IMU_DATA = struct.calcsize(FMT_IMU_DATA)

# LidarVersionData: 4B hw + 4B sw + 24B name + 8B date + 40B reserve = 80 bytes
FMT_VERSION_DATA = "<4s4s24s8s40s"
SIZE_VERSION_DATA = 80

# UserCtrlCmd: uint32 cmd_type + uint32 cmd_value = 8 bytes
FMT_USER_CMD = "<II"
SIZE_USER_CMD = 8

# AckData: 4 uint32 = 16 bytes
FMT_ACK_DATA = "<IIII"
SIZE_ACK_DATA = 16

# WorkModeConfig: uint32 = 4 bytes
FMT_WORK_MODE = "<I"
SIZE_WORK_MODE = 4

# IpAddressConfig: 4x4B + 2 uint16 = 20 bytes
FMT_IP_CONFIG = "<4s4s4s4sHH"
SIZE_IP_CONFIG = 20

# MacAddressConfig: 6B + 2B reserve = 8 bytes
FMT_MAC_CONFIG = "<6s2s"
SIZE_MAC_CONFIG = 8

# ─── Struct sizes for point data payloads ────────────────────────────────────

# LidarPointData fixed header before arrays:
# DataInfo(16) + InsideState(36) + CalibParam(32) + 9 floats(36) + 1 uint32(4) = 124
_POINT_DATA_HEADER_SIZE = SIZE_DATA_INFO + SIZE_INSIDE_STATE + SIZE_CALIB_PARAM + 9 * 4 + 4
# 300 uint16 ranges + 300 uint8 intensities = 900
_POINT_DATA_ARRAYS_SIZE = 300 * 2 + 300 * 1
# Total LidarPointData = 1024 bytes (but protocol says 1012 — we compute from struct)
POINT_DATA_3D_SIZE = _POINT_DATA_HEADER_SIZE + _POINT_DATA_ARRAYS_SIZE  # 1024

# Lidar2DPointData:
# DataInfo(16) + InsideState(36) + CalibParam(32) + 6 floats(24) + 1 uint32(4) = 112
_POINT_2D_HEADER_SIZE = SIZE_DATA_INFO + SIZE_INSIDE_STATE + SIZE_CALIB_PARAM + 6 * 4 + 4
# 1800 uint16 ranges + 1800 uint8 intensities = 5400
_POINT_2D_ARRAYS_SIZE = 1800 * 2 + 1800 * 1
POINT_DATA_2D_SIZE = _POINT_2D_HEADER_SIZE + _POINT_2D_ARRAYS_SIZE

MAX_POINTS_3D = 300
MAX_POINTS_2D = 1800


# ─── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class FrameHeader:
    magic: bytes
    packet_type: int
    packet_size: int

    @classmethod
    def from_bytes(cls, data: bytes) -> FrameHeader:
        magic, ptype, psize = struct.unpack_from(FMT_FRAME_HEADER, data)
        return cls(magic=magic, packet_type=ptype, packet_size=psize)

    def to_bytes(self) -> bytes:
        return struct.pack(FMT_FRAME_HEADER, self.magic, self.packet_type, self.packet_size)


@dataclass
class FrameTail:
    crc32: int
    msg_type_check: int
    reserve: bytes = b"\x00\x00"
    tail: bytes = FRAME_TAIL_MAGIC

    @classmethod
    def from_bytes(cls, data: bytes) -> FrameTail:
        crc, msg_check, reserve, tail = struct.unpack_from(FMT_FRAME_TAIL, data)
        return cls(crc32=crc, msg_type_check=msg_check, reserve=reserve, tail=tail)

    def to_bytes(self) -> bytes:
        return struct.pack(FMT_FRAME_TAIL, self.crc32, self.msg_type_check,
                           self.reserve, self.tail)


@dataclass
class CalibParam:
    """LidarCalibParam from protocol."""

    a_axis_dist: float = 0.0
    b_axis_dist: float = 0.0
    theta_angle_bias: float = 0.0
    alpha_angle_bias: float = 0.0
    beta_angle: float = 0.0
    xi_angle: float = 0.0
    range_bias: float = 0.0
    range_scale: float = 1.0

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> CalibParam:
        vals = struct.unpack_from(FMT_CALIB_PARAM, data, offset)
        return cls(*vals)


@dataclass
class DataInfo:
    """Packet sequence/size/timestamp info."""

    seq: int = 0
    payload_size: int = 0
    stamp_sec: int = 0
    stamp_nsec: int = 0

    @property
    def stamp(self) -> float:
        return self.stamp_sec + self.stamp_nsec / 1.0e9

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> DataInfo:
        seq, ps, sec, nsec = struct.unpack_from(FMT_DATA_INFO, data, offset)
        return cls(seq=seq, payload_size=ps, stamp_sec=sec, stamp_nsec=nsec)


@dataclass
class PointData3D:
    """Parsed 3D point data packet payload."""

    info: DataInfo
    calib: CalibParam
    com_horizontal_angle_start: float = 0.0
    com_horizontal_angle_step: float = 0.0
    scan_period: float = 0.0
    range_min: float = 0.0
    range_max: float = 50.0
    angle_min: float = 0.0
    angle_increment: float = 0.0
    time_increment: float = 0.0
    point_num: int = 0
    ranges: np.ndarray = None  # uint16[300]
    intensities: np.ndarray = None  # uint8[300]
    # Inside state (kept for dirty index etc.)
    dirty_index: float = 0.0

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> PointData3D:
        o = offset

        # DataInfo (16 bytes)
        info = DataInfo.from_bytes(data, o)
        o += SIZE_DATA_INFO

        # InsideState (36 bytes) — extract dirty_index
        state_vals = struct.unpack_from(FMT_INSIDE_STATE, data, o)
        dirty_index = state_vals[2]  # dirty_index is 3rd field
        o += SIZE_INSIDE_STATE

        # CalibParam (32 bytes)
        calib = CalibParam.from_bytes(data, o)
        o += SIZE_CALIB_PARAM

        # 9 floats + 1 uint32
        scan_fields = struct.unpack_from("<9fI", data, o)
        o += 9 * 4 + 4

        # Ranges: 300 uint16
        ranges = np.frombuffer(data, dtype=np.uint16, count=MAX_POINTS_3D, offset=o).copy()
        o += MAX_POINTS_3D * 2

        # Intensities: 300 uint8
        intensities = np.frombuffer(data, dtype=np.uint8, count=MAX_POINTS_3D, offset=o).copy()

        return cls(
            info=info,
            calib=calib,
            com_horizontal_angle_start=scan_fields[0],
            com_horizontal_angle_step=scan_fields[1],
            scan_period=scan_fields[2],
            range_min=scan_fields[3],
            range_max=scan_fields[4],
            angle_min=scan_fields[5],
            angle_increment=scan_fields[6],
            time_increment=scan_fields[7],
            point_num=scan_fields[9],  # uint32 after 9 floats (index 8 is last float)
            ranges=ranges,
            intensities=intensities,
            dirty_index=dirty_index,
        )


@dataclass
class PointData2D:
    """Parsed 2D point data packet payload."""

    info: DataInfo
    calib: CalibParam
    scan_period: float = 0.0
    range_min: float = 0.0
    range_max: float = 50.0
    angle_min: float = 0.0
    angle_increment: float = 0.0
    time_increment: float = 0.0
    point_num: int = 0
    ranges: np.ndarray = None  # uint16[1800]
    intensities: np.ndarray = None  # uint8[1800]
    dirty_index: float = 0.0

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> PointData2D:
        o = offset

        # DataInfo (16 bytes)
        info = DataInfo.from_bytes(data, o)
        o += SIZE_DATA_INFO

        # InsideState (36 bytes)
        state_vals = struct.unpack_from(FMT_INSIDE_STATE, data, o)
        dirty_index = state_vals[2]
        o += SIZE_INSIDE_STATE

        # CalibParam (32 bytes)
        calib = CalibParam.from_bytes(data, o)
        o += SIZE_CALIB_PARAM

        # 6 floats + 1 uint32
        scan_fields = struct.unpack_from("<6fI", data, o)
        o += 6 * 4 + 4

        # Ranges: 1800 uint16
        ranges = np.frombuffer(data, dtype=np.uint16, count=MAX_POINTS_2D, offset=o).copy()
        o += MAX_POINTS_2D * 2

        # Intensities: 1800 uint8
        intensities = np.frombuffer(data, dtype=np.uint8, count=MAX_POINTS_2D, offset=o).copy()

        return cls(
            info=info,
            calib=calib,
            scan_period=scan_fields[0],
            range_min=scan_fields[1],
            range_max=scan_fields[2],
            angle_min=scan_fields[3],
            angle_increment=scan_fields[4],
            time_increment=scan_fields[5],
            point_num=scan_fields[6],
            ranges=ranges,
            intensities=intensities,
            dirty_index=dirty_index,
        )


@dataclass
class ImuDataPacket:
    """Parsed IMU data payload."""

    info: DataInfo
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> ImuDataPacket:
        vals = struct.unpack_from(FMT_IMU_DATA, data, offset)
        info = DataInfo(seq=vals[0], payload_size=vals[1],
                        stamp_sec=vals[2], stamp_nsec=vals[3])
        quat = (vals[4], vals[5], vals[6], vals[7])
        gyro = (vals[8], vals[9], vals[10])
        accel = (vals[11], vals[12], vals[13])
        return cls(info=info, quaternion=quat, angular_velocity=gyro,
                   linear_acceleration=accel)


@dataclass
class VersionDataPacket:
    """Parsed version data payload."""

    hw_version: str = ""
    sw_version: str = ""
    name: str = ""
    date: str = ""

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> VersionDataPacket:
        hw, sw, name, date, _ = struct.unpack_from(FMT_VERSION_DATA, data, offset)

        def _ver_str(b: bytes) -> str:
            return ".".join(str(x) for x in b)

        return cls(
            hw_version=_ver_str(hw),
            sw_version=_ver_str(sw),
            name=name.rstrip(b"\x00").decode("ascii", errors="replace"),
            date=date.rstrip(b"\x00").decode("ascii", errors="replace"),
        )


@dataclass
class AckDataPacket:
    """Parsed ACK payload."""

    packet_type: int = 0
    cmd_type: int = 0
    cmd_value: int = 0
    status: int = 0

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> AckDataPacket:
        vals = struct.unpack_from(FMT_ACK_DATA, data, offset)
        return cls(*vals)


# ─── Command packet builders ────────────────────────────────────────────────

def build_user_cmd_packet(cmd_type: int, cmd_value: int = 0) -> bytes:
    """Build a complete framed packet for a user control command."""
    from unitree_sdk2_mlx.crc import crc32 as compute_crc

    payload = struct.pack(FMT_USER_CMD, cmd_type, cmd_value)
    total_size = SIZE_FRAME_HEADER + len(payload) + SIZE_FRAME_TAIL

    header = struct.pack(FMT_FRAME_HEADER, FRAME_HEADER_MAGIC, PACKET_USER_CMD, total_size)
    crc_data = header + payload
    crc_val = compute_crc(crc_data)

    tail = struct.pack(FMT_FRAME_TAIL, crc_val, PACKET_USER_CMD, b"\x00\x00", FRAME_TAIL_MAGIC)

    return header + payload + tail


def build_work_mode_packet(mode: int) -> bytes:
    """Build a work mode configuration packet."""
    from unitree_sdk2_mlx.crc import crc32 as compute_crc

    payload = struct.pack(FMT_WORK_MODE, mode)
    total_size = SIZE_FRAME_HEADER + len(payload) + SIZE_FRAME_TAIL

    header = struct.pack(FMT_FRAME_HEADER, FRAME_HEADER_MAGIC, PACKET_WORK_MODE, total_size)
    crc_data = header + payload
    crc_val = compute_crc(crc_data)

    tail = struct.pack(FMT_FRAME_TAIL, crc_val, PACKET_WORK_MODE, b"\x00\x00", FRAME_TAIL_MAGIC)

    return header + payload + tail


def build_ip_config_packet(
    lidar_ip: tuple[int, ...],
    user_ip: tuple[int, ...],
    gateway: tuple[int, ...],
    subnet_mask: tuple[int, ...],
    lidar_port: int,
    user_port: int,
) -> bytes:
    """Build an IP address configuration packet."""
    from unitree_sdk2_mlx.crc import crc32 as compute_crc

    payload = struct.pack(
        FMT_IP_CONFIG,
        bytes(lidar_ip), bytes(user_ip), bytes(gateway), bytes(subnet_mask),
        lidar_port, user_port,
    )
    total_size = SIZE_FRAME_HEADER + len(payload) + SIZE_FRAME_TAIL

    header = struct.pack(FMT_FRAME_HEADER, FRAME_HEADER_MAGIC, PACKET_IP_CONFIG, total_size)
    crc_data = header + payload
    crc_val = compute_crc(crc_data)

    tail = struct.pack(FMT_FRAME_TAIL, crc_val, PACKET_IP_CONFIG, b"\x00\x00", FRAME_TAIL_MAGIC)

    return header + payload + tail


def build_timestamp_packet(sec: int, nsec: int) -> bytes:
    """Build a timestamp sync packet."""
    from unitree_sdk2_mlx.crc import crc32 as compute_crc

    payload = struct.pack(FMT_TIMESTAMP, sec, nsec)
    total_size = SIZE_FRAME_HEADER + len(payload) + SIZE_FRAME_TAIL

    header = struct.pack(FMT_FRAME_HEADER, FRAME_HEADER_MAGIC, PACKET_TIMESTAMP, total_size)
    crc_data = header + payload
    crc_val = compute_crc(crc_data)

    tail = struct.pack(FMT_FRAME_TAIL, crc_val, PACKET_TIMESTAMP, b"\x00\x00", FRAME_TAIL_MAGIC)

    return header + payload + tail
