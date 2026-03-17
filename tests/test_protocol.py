"""Tests for protocol structures."""

import struct

import numpy as np

from unitree_sdk2_mlx.protocol import (
    FrameHeader,
    FrameTail,
    CalibParam,
    DataInfo,
    ImuDataPacket,
    VersionDataPacket,
    AckDataPacket,
    FRAME_HEADER_MAGIC,
    FRAME_TAIL_MAGIC,
    SIZE_FRAME_HEADER,
    SIZE_FRAME_TAIL,
    FMT_FRAME_HEADER,
    FMT_FRAME_TAIL,
    FMT_CALIB_PARAM,
    FMT_DATA_INFO,
    FMT_IMU_DATA,
    PACKET_USER_CMD,
    build_user_cmd_packet,
    build_work_mode_packet,
    build_timestamp_packet,
)
from unitree_sdk2_mlx.crc import crc32


def test_frame_header_size():
    assert SIZE_FRAME_HEADER == 12


def test_frame_tail_size():
    assert SIZE_FRAME_TAIL == 12


def test_frame_header_roundtrip():
    hdr = FrameHeader(magic=FRAME_HEADER_MAGIC, packet_type=102, packet_size=1036)
    data = hdr.to_bytes()
    assert len(data) == SIZE_FRAME_HEADER
    hdr2 = FrameHeader.from_bytes(data)
    assert hdr2.magic == FRAME_HEADER_MAGIC
    assert hdr2.packet_type == 102
    assert hdr2.packet_size == 1036


def test_frame_tail_roundtrip():
    tail = FrameTail(crc32=0xDEADBEEF, msg_type_check=102)
    data = tail.to_bytes()
    assert len(data) == SIZE_FRAME_TAIL
    tail2 = FrameTail.from_bytes(data)
    assert tail2.crc32 == 0xDEADBEEF
    assert tail2.msg_type_check == 102
    assert tail2.tail == FRAME_TAIL_MAGIC


def test_calib_param_from_bytes():
    vals = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0)
    data = struct.pack(FMT_CALIB_PARAM, *vals)
    calib = CalibParam.from_bytes(data)
    assert abs(calib.a_axis_dist - 0.1) < 1e-6
    assert abs(calib.range_scale - 1.0) < 1e-6


def test_data_info_from_bytes():
    data = struct.pack(FMT_DATA_INFO, 42, 1024, 1000, 500000000)
    info = DataInfo.from_bytes(data)
    assert info.seq == 42
    assert info.payload_size == 1024
    assert abs(info.stamp - 1000.5) < 1e-6


def test_imu_data_from_bytes():
    vals = (1, 56, 100, 200000000,  # DataInfo
            0.0, 0.0, 0.0, 1.0,    # quaternion
            0.1, 0.2, 0.3,         # angular velocity
            9.8, 0.0, 0.0)         # linear acceleration
    data = struct.pack(FMT_IMU_DATA, *vals)
    imu = ImuDataPacket.from_bytes(data)
    assert imu.info.seq == 1
    assert abs(imu.quaternion[3] - 1.0) < 1e-6
    assert abs(imu.linear_acceleration[0] - 9.8) < 1e-5


def test_build_user_cmd_packet():
    pkt = build_user_cmd_packet(cmd_type=1, cmd_value=0)
    assert pkt[:4] == FRAME_HEADER_MAGIC
    assert pkt[-2:] == FRAME_TAIL_MAGIC
    # Verify CRC
    payload_end = len(pkt) - SIZE_FRAME_TAIL
    crc_data = pkt[:payload_end]
    crc_in_packet = struct.unpack_from("<I", pkt, payload_end)[0]
    assert crc32(crc_data) == crc_in_packet


def test_build_work_mode_packet():
    pkt = build_work_mode_packet(mode=0)
    assert pkt[:4] == FRAME_HEADER_MAGIC
    assert pkt[-2:] == FRAME_TAIL_MAGIC


def test_build_timestamp_packet():
    pkt = build_timestamp_packet(sec=1000, nsec=500000000)
    assert pkt[:4] == FRAME_HEADER_MAGIC
    assert pkt[-2:] == FRAME_TAIL_MAGIC
