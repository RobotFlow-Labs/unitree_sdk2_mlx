"""Tests for CRC32 implementation."""

import zlib

from unitree_sdk2_mlx.crc import crc32


def test_crc32_empty():
    assert crc32(b"") == zlib.crc32(b"") & 0xFFFFFFFF


def test_crc32_hello():
    expected = zlib.crc32(b"hello") & 0xFFFFFFFF
    assert crc32(b"hello") == expected


def test_crc32_known_value():
    # CRC32 of "123456789" is 0xCBF43926
    assert crc32(b"123456789") == 0xCBF43926


def test_crc32_binary_data():
    data = bytes(range(256))
    assert crc32(data) == zlib.crc32(data) & 0xFFFFFFFF


def test_crc32_frame_header_magic():
    """Test with the actual frame header magic bytes."""
    data = bytes([0x55, 0xAA, 0x05, 0x0A])
    assert crc32(data) == zlib.crc32(data) & 0xFFFFFFFF
