"""Tests for frame parser."""

import struct

import numpy as np

from unitree_sdk2_mlx.crc import crc32
from unitree_sdk2_mlx.parser import FrameParser
from unitree_sdk2_mlx.protocol import (
    FRAME_HEADER_MAGIC,
    FRAME_TAIL_MAGIC,
    PACKET_IMU,
    PACKET_ACK,
    SIZE_FRAME_HEADER,
    SIZE_FRAME_TAIL,
    FMT_FRAME_HEADER,
    FMT_FRAME_TAIL,
    FMT_IMU_DATA,
    FMT_ACK_DATA,
)


def _build_test_packet(packet_type: int, payload: bytes) -> bytes:
    """Build a valid framed packet for testing."""
    total = SIZE_FRAME_HEADER + len(payload) + SIZE_FRAME_TAIL
    header = struct.pack(FMT_FRAME_HEADER, FRAME_HEADER_MAGIC, packet_type, total)
    crc_data = header + payload
    crc_val = crc32(crc_data)
    tail = struct.pack(FMT_FRAME_TAIL, crc_val, packet_type, b"\x00\x00", FRAME_TAIL_MAGIC)
    return header + payload + tail


def test_parse_imu_packet():
    """Test parsing a complete IMU packet."""
    imu_vals = (1, 56, 100, 200000000, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 9.8, 0.0, 0.0)
    payload = struct.pack(FMT_IMU_DATA, *imu_vals)
    packet_bytes = _build_test_packet(PACKET_IMU, payload)

    parser = FrameParser()
    parser.feed(packet_bytes)
    result = parser.parse_one()

    assert result is not None
    assert result.packet_type == PACKET_IMU
    assert abs(result.payload.quaternion[3] - 1.0) < 1e-6


def test_parse_ack_packet():
    """Test parsing an ACK packet."""
    payload = struct.pack(FMT_ACK_DATA, 100, 1, 0, 1)
    packet_bytes = _build_test_packet(PACKET_ACK, payload)

    parser = FrameParser()
    parser.feed(packet_bytes)
    result = parser.parse_one()

    assert result is not None
    assert result.packet_type == PACKET_ACK
    assert result.payload.status == 1


def test_parse_with_leading_garbage():
    """Parser should skip garbage bytes before a valid frame."""
    imu_vals = (1, 56, 100, 200000000, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 9.8, 0.0, 0.0)
    payload = struct.pack(FMT_IMU_DATA, *imu_vals)
    packet_bytes = _build_test_packet(PACKET_IMU, payload)

    garbage = b"\xDE\xAD\xBE\xEF\x01\x02\x03"
    parser = FrameParser()
    parser.feed(garbage + packet_bytes)
    result = parser.parse_one()

    assert result is not None
    assert result.packet_type == PACKET_IMU


def test_parse_two_packets():
    """Parse two consecutive packets."""
    imu_vals = (1, 56, 100, 200000000, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 9.8, 0.0, 0.0)
    payload = struct.pack(FMT_IMU_DATA, *imu_vals)
    pkt1 = _build_test_packet(PACKET_IMU, payload)
    pkt2 = _build_test_packet(PACKET_IMU, payload)

    parser = FrameParser()
    parser.feed(pkt1 + pkt2)

    r1 = parser.parse_one()
    r2 = parser.parse_one()

    assert r1 is not None
    assert r2 is not None
    assert r1.packet_type == PACKET_IMU
    assert r2.packet_type == PACKET_IMU


def test_parse_incomplete_packet():
    """Return None if packet is incomplete."""
    imu_vals = (1, 56, 100, 200000000, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 9.8, 0.0, 0.0)
    payload = struct.pack(FMT_IMU_DATA, *imu_vals)
    packet_bytes = _build_test_packet(PACKET_IMU, payload)

    parser = FrameParser()
    # Feed only half the packet
    parser.feed(packet_bytes[:len(packet_bytes) // 2])
    result = parser.parse_one()
    assert result is None

    # Feed the rest
    parser.feed(packet_bytes[len(packet_bytes) // 2:])
    result = parser.parse_one()
    assert result is not None
    assert result.packet_type == PACKET_IMU


def test_parse_corrupted_crc():
    """Packet with wrong CRC should be skipped."""
    imu_vals = (1, 56, 100, 200000000, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 9.8, 0.0, 0.0)
    payload = struct.pack(FMT_IMU_DATA, *imu_vals)
    packet_bytes = bytearray(_build_test_packet(PACKET_IMU, payload))
    # Corrupt the CRC (first 4 bytes of tail)
    crc_offset = len(packet_bytes) - SIZE_FRAME_TAIL
    packet_bytes[crc_offset] ^= 0xFF

    parser = FrameParser()
    parser.feed(bytes(packet_bytes))
    result = parser.parse_one()
    assert result is None


def test_clear_buffer():
    parser = FrameParser()
    parser.feed(b"\x00" * 100)
    assert parser.buffer_size > 0
    parser.clear()
    assert parser.buffer_size == 0
