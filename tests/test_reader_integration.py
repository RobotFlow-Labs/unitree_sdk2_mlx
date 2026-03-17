"""Integration tests for UnitreeLidarReader end-to-end pipeline.

Feeds raw binary packets through the full stack:
transport -> parser -> transforms -> reader output.
"""

from __future__ import annotations

import numpy as np

from unitree_sdk2_mlx.reader import UnitreeLidarReader
from unitree_sdk2_mlx.protocol import PACKET_POINT_3D, PACKET_IMU, PACKET_VERSION
from conftest import (
    build_raw_3d_payload,
    build_raw_imu_payload,
    build_raw_version_payload,
    frame_packet,
)


class FakeTransport:
    """In-memory transport that feeds pre-built packet data."""

    def __init__(self):
        self._buffer = bytearray()
        self._open = True

    def inject(self, data: bytes) -> None:
        self._buffer.extend(data)

    def open(self) -> int:
        self._open = True
        return 0

    def close(self) -> None:
        self._open = False

    def read(self, max_bytes: int = 8192) -> bytes:
        if not self._buffer:
            return b""
        chunk = bytes(self._buffer[:max_bytes])
        del self._buffer[:max_bytes]
        return chunk

    def write(self, data: bytes) -> int:
        return len(data)

    @property
    def is_open(self) -> bool:
        return self._open


def _make_reader_with_transport() -> tuple[UnitreeLidarReader, FakeTransport]:
    """Create a reader with an injected fake transport."""
    reader = UnitreeLidarReader()
    transport = FakeTransport()
    reader._transport = transport
    return reader, transport


class TestReaderIMU:
    def test_parse_imu(self):
        reader, transport = _make_reader_with_transport()
        pkt = frame_packet(PACKET_IMU, build_raw_imu_payload(seq=42))
        transport.inject(pkt)

        result = reader.run_parse()
        assert result == PACKET_IMU

        imu = reader.get_imu_data()
        assert imu is not None
        assert imu.seq == 42
        assert abs(imu.quaternion[3] - 1.0) < 1e-6
        assert abs(imu.linear_acceleration[0] - 9.8) < 1e-4

    def test_imu_not_ready_before_parse(self):
        reader, _ = _make_reader_with_transport()
        assert reader.get_imu_data() is None


class TestReaderVersion:
    def test_parse_version(self):
        reader, transport = _make_reader_with_transport()
        pkt = frame_packet(PACKET_VERSION, build_raw_version_payload())
        transport.inject(pkt)

        result = reader.run_parse()
        assert result == PACKET_VERSION

        assert reader.get_version_firmware() == "2.3.3.0"
        assert reader.get_version_hardware() == "1.1.1.1"


class TestReaderPointCloud:
    def test_cloud_accumulation(self):
        """Cloud is only ready after cloud_scan_num (default 18) packets."""
        reader, transport = _make_reader_with_transport()
        reader._cloud_scan_num = 3  # reduce for test speed

        # Feed 2 packets — cloud should NOT be ready
        for i in range(2):
            pkt = frame_packet(PACKET_POINT_3D,
                               build_raw_3d_payload(seq=i, point_num=10, range_val=3000))
            transport.inject(pkt)
            reader.run_parse()

        assert reader.get_point_cloud() is None

        # Feed 3rd packet — cloud should be ready
        pkt = frame_packet(PACKET_POINT_3D,
                           build_raw_3d_payload(seq=2, point_num=10, range_val=3000))
        transport.inject(pkt)
        reader.run_parse()

        cloud = reader.get_point_cloud()
        assert cloud is not None
        assert cloud.size > 0
        assert cloud.points.shape[1] == 6
        # All coordinates should be finite
        assert np.all(np.isfinite(cloud.points[:, :3]))

    def test_cloud_consumed_once(self):
        """get_point_cloud returns None on second call (already consumed)."""
        reader, transport = _make_reader_with_transport()
        reader._cloud_scan_num = 1

        pkt = frame_packet(PACKET_POINT_3D,
                           build_raw_3d_payload(point_num=10, range_val=5000))
        transport.inject(pkt)
        reader.run_parse()

        assert reader.get_point_cloud() is not None
        assert reader.get_point_cloud() is None

    def test_dirty_percentage_tracked(self):
        reader, transport = _make_reader_with_transport()
        reader._cloud_scan_num = 1

        pkt = frame_packet(PACKET_POINT_3D,
                           build_raw_3d_payload(point_num=5, range_val=2000))
        transport.inject(pkt)
        reader.run_parse()

        # dirty_index was set to 3.0 in the fixture
        dp = reader.get_dirty_percentage()
        assert dp is not None
        assert abs(dp - 3.0) < 1e-5


class TestReaderContextManager:
    def test_context_manager(self):
        with UnitreeLidarReader() as reader:
            transport = FakeTransport()
            reader._transport = transport
            assert reader.is_connected
        # After exit, transport should be closed
        assert not reader.is_connected


class TestReaderMultiPacket:
    def test_interleaved_imu_and_points(self):
        """Parse interleaved IMU and point packets."""
        reader, transport = _make_reader_with_transport()
        reader._cloud_scan_num = 2

        # IMU, point, IMU, point
        transport.inject(frame_packet(PACKET_IMU, build_raw_imu_payload(seq=1)))
        transport.inject(frame_packet(PACKET_POINT_3D,
                                      build_raw_3d_payload(seq=1, point_num=10, range_val=4000)))
        transport.inject(frame_packet(PACKET_IMU, build_raw_imu_payload(seq=2)))
        transport.inject(frame_packet(PACKET_POINT_3D,
                                      build_raw_3d_payload(seq=2, point_num=10, range_val=4000)))

        parsed_types = []
        for _ in range(10):  # parse until buffer drained
            result = reader.run_parse()
            if result:
                parsed_types.append(result)

        assert PACKET_IMU in parsed_types
        assert PACKET_POINT_3D in parsed_types

        # IMU should have been parsed
        imu = reader.get_imu_data()
        assert imu is not None

        # Cloud should be ready (2 point packets = cloud_scan_num)
        cloud = reader.get_point_cloud()
        assert cloud is not None
        assert cloud.size > 0
