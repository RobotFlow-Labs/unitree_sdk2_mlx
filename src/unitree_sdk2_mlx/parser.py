"""Frame parser — state machine that extracts packets from a byte stream.

Reimplements the closed-source parsing logic from libunilidar_sdk2.a.
"""

from __future__ import annotations

import struct
from enum import IntEnum, auto
from typing import Optional

from unitree_sdk2_mlx.crc import crc32
from unitree_sdk2_mlx.protocol import (
    FRAME_HEADER_MAGIC,
    FRAME_TAIL_MAGIC,
    SIZE_FRAME_HEADER,
    SIZE_FRAME_TAIL,
    FMT_FRAME_HEADER,
    FMT_FRAME_TAIL,
    PACKET_POINT_3D,
    PACKET_POINT_2D,
    PACKET_IMU,
    PACKET_VERSION,
    PACKET_ACK,
    PACKET_TIMESTAMP,
    PointData3D,
    PointData2D,
    ImuDataPacket,
    VersionDataPacket,
    AckDataPacket,
)


class _State(IntEnum):
    SEEKING_HEADER = auto()
    READING_FRAME = auto()


class ParsedPacket:
    """Container for a parsed packet."""

    __slots__ = ("packet_type", "payload")

    def __init__(self, packet_type: int, payload: object):
        self.packet_type = packet_type
        self.payload = payload


class FrameParser:
    """State-machine parser for the Unitree LiDAR binary protocol.

    Usage:
        parser = FrameParser()
        parser.feed(raw_bytes)
        packet = parser.parse_one()
        if packet is not None:
            print(packet.packet_type, packet.payload)
    """

    def __init__(self, max_buffer: int = 65536):
        self._buffer = bytearray()
        self._max_buffer = max_buffer
        self._state = _State.SEEKING_HEADER

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def feed(self, data: bytes) -> None:
        """Append raw bytes from transport into the internal buffer."""
        self._buffer.extend(data)
        # Prevent unbounded growth
        if len(self._buffer) > self._max_buffer:
            self._buffer = self._buffer[-self._max_buffer:]

    def parse_one(self) -> Optional[ParsedPacket]:
        """Try to parse one complete packet from the buffer.

        Returns None if no complete packet is available.
        """
        while True:
            if self._state == _State.SEEKING_HEADER:
                idx = self._buffer.find(FRAME_HEADER_MAGIC)
                if idx < 0:
                    # Keep last 3 bytes in case header spans a feed boundary
                    if len(self._buffer) > 3:
                        self._buffer = self._buffer[-3:]
                    return None
                # Discard bytes before the header
                if idx > 0:
                    del self._buffer[:idx]
                self._state = _State.READING_FRAME

            if self._state == _State.READING_FRAME:
                # Need at least the frame header to know total size
                if len(self._buffer) < SIZE_FRAME_HEADER:
                    return None

                _, packet_type, packet_size = struct.unpack_from(
                    FMT_FRAME_HEADER, self._buffer
                )

                # Sanity check packet size
                if packet_size < SIZE_FRAME_HEADER + SIZE_FRAME_TAIL or packet_size > 16384:
                    # Invalid — skip this header and re-seek
                    del self._buffer[:4]
                    self._state = _State.SEEKING_HEADER
                    continue

                # Wait for full packet
                if len(self._buffer) < packet_size:
                    return None

                # Extract the full frame
                frame = bytes(self._buffer[:packet_size])
                del self._buffer[:packet_size]
                self._state = _State.SEEKING_HEADER

                # Verify CRC
                payload_end = packet_size - SIZE_FRAME_TAIL
                crc_data = frame[:payload_end]
                tail_data = frame[payload_end:]

                crc_val, msg_type_check, reserve, tail_magic = struct.unpack_from(
                    FMT_FRAME_TAIL, tail_data
                )

                if tail_magic != FRAME_TAIL_MAGIC:
                    continue  # Bad tail, skip

                computed_crc = crc32(crc_data)
                if computed_crc != crc_val:
                    continue  # CRC mismatch, skip

                # Parse payload based on packet type
                payload_data = frame[SIZE_FRAME_HEADER:payload_end]
                parsed = self._dispatch(packet_type, payload_data)
                if parsed is not None:
                    return ParsedPacket(packet_type=packet_type, payload=parsed)

                # Unknown packet type — continue seeking
                continue

        return None  # unreachable but keeps type checker happy

    def _dispatch(self, packet_type: int, data: bytes) -> Optional[object]:
        """Dispatch payload parsing based on packet type."""
        try:
            if packet_type == PACKET_POINT_3D:
                return PointData3D.from_bytes(data)
            elif packet_type == PACKET_POINT_2D:
                return PointData2D.from_bytes(data)
            elif packet_type == PACKET_IMU:
                return ImuDataPacket.from_bytes(data)
            elif packet_type == PACKET_VERSION:
                return VersionDataPacket.from_bytes(data)
            elif packet_type == PACKET_ACK:
                return AckDataPacket.from_bytes(data)
            else:
                # Timestamp, config acks, etc. — return raw bytes
                return data
        except (struct.error, IndexError, ValueError):
            return None

    def clear(self) -> None:
        """Clear the internal buffer."""
        self._buffer.clear()
        self._state = _State.SEEKING_HEADER
