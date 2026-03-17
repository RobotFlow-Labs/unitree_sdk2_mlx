"""High-level LiDAR reader — mirrors UnitreeLidarReader from the C++ SDK.

This is the main entry point for users of the library.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from unitree_sdk2_mlx.parser import FrameParser, ParsedPacket
from unitree_sdk2_mlx.protocol import (
    PACKET_POINT_3D,
    PACKET_POINT_2D,
    PACKET_IMU,
    PACKET_VERSION,
    PACKET_ACK,
    USER_CMD_RESET,
    USER_CMD_STANDBY,
    USER_CMD_VERSION_GET,
    USER_CMD_LATENCY,
    PointData3D,
    PointData2D,
    ImuDataPacket,
    VersionDataPacket,
    AckDataPacket,
    build_user_cmd_packet,
    build_work_mode_packet,
    build_ip_config_packet,
    build_timestamp_packet,
)
from unitree_sdk2_mlx.transforms import transform_3d, transform_2d
from unitree_sdk2_mlx.types import PointCloudUnitree, ImuData, VersionInfo
from unitree_sdk2_mlx.transport.base import Transport

# SDK version
SDK_VERSION = "0.1.0"


class UnitreeLidarReader:
    """Python equivalent of C++ UnitreeLidarReader.

    Usage:
        reader = UnitreeLidarReader()
        reader.initialize_udp()
        reader.start_rotation()

        while True:
            ptype = reader.run_parse()
            if ptype == PACKET_POINT_3D:
                cloud = reader.get_point_cloud()
                if cloud is not None:
                    print(f"Cloud: {cloud.size} points")
            elif ptype == PACKET_IMU:
                imu = reader.get_imu_data()
                if imu is not None:
                    print(f"IMU: {imu.quaternion}")
    """

    def __init__(self):
        self._parser = FrameParser()
        self._transport: Optional[Transport] = None

        # Configuration
        self._cloud_scan_num: int = 18
        self._use_system_timestamp: bool = True
        self._range_min: float = 0.0
        self._range_max: float = 100.0

        # Accumulated state
        self._point_packets: list[PointData3D] = []
        self._point_2d_packets: list[PointData2D] = []
        self._cloud: Optional[PointCloudUnitree] = None
        self._cloud_ready: bool = False

        self._imu: Optional[ImuData] = None
        self._imu_ready: bool = False

        self._version: Optional[VersionInfo] = None
        self._version_ready: bool = False

        self._time_delay: Optional[float] = None
        self._dirty_percentage: Optional[float] = None

        self._last_ack: Optional[AckDataPacket] = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ── Initialization ───────────────────────────────────────────────────

    def initialize_udp(
        self,
        lidar_port: int = 6101,
        lidar_ip: str = "192.168.1.62",
        local_port: int = 6201,
        local_ip: str = "192.168.1.2",
        cloud_scan_num: int = 18,
        use_system_timestamp: bool = True,
        range_min: float = 0.0,
        range_max: float = 100.0,
    ) -> int:
        """Initialize UDP connection. Returns 0 on success, -1 on failure."""
        from unitree_sdk2_mlx.transport.udp import UDPTransport

        self._cloud_scan_num = cloud_scan_num
        self._use_system_timestamp = use_system_timestamp
        self._range_min = range_min
        self._range_max = range_max

        transport = UDPTransport(
            lidar_port=lidar_port,
            lidar_ip=lidar_ip,
            local_port=local_port,
            local_ip=local_ip,
        )
        result = transport.open()
        if result == 0:
            self._transport = transport
        return result

    def initialize_serial(
        self,
        port: Optional[str] = None,
        baudrate: int = 4_000_000,
        cloud_scan_num: int = 18,
        use_system_timestamp: bool = True,
        range_min: float = 0.0,
        range_max: float = 100.0,
    ) -> int:
        """Initialize serial connection. Returns 0 on success, -1 on failure."""
        from unitree_sdk2_mlx.transport.serial import SerialTransport

        self._cloud_scan_num = cloud_scan_num
        self._use_system_timestamp = use_system_timestamp
        self._range_min = range_min
        self._range_max = range_max

        transport = SerialTransport(port=port, baudrate=baudrate)
        result = transport.open()
        if result == 0:
            self._transport = transport
        return result

    def close(self) -> None:
        """Close the connection."""
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    # ── Main parse loop ──────────────────────────────────────────────────

    def run_parse(self) -> int:
        """Read from transport and try to parse one packet.

        Returns the packet type ID, or 0 if nothing parsed.
        """
        if self._transport is None:
            return 0

        data = self._transport.read()
        if data:
            self._parser.feed(data)

        packet = self._parser.parse_one()
        if packet is None:
            return 0

        self._dispatch(packet)
        return packet.packet_type

    def _dispatch(self, packet: ParsedPacket) -> None:
        """Handle a parsed packet."""
        if packet.packet_type == PACKET_POINT_3D:
            self._handle_point_3d(packet.payload)
        elif packet.packet_type == PACKET_POINT_2D:
            self._handle_point_2d(packet.payload)
        elif packet.packet_type == PACKET_IMU:
            self._handle_imu(packet.payload)
        elif packet.packet_type == PACKET_VERSION:
            self._handle_version(packet.payload)
        elif packet.packet_type == PACKET_ACK:
            self._last_ack = packet.payload

    def _handle_point_3d(self, data: PointData3D) -> None:
        """Accumulate 3D point packets and assemble cloud."""
        # Track dirty percentage from inside state
        if data.dirty_index > 0:
            self._dirty_percentage = data.dirty_index

        self._point_packets.append(data)

        if len(self._point_packets) >= self._cloud_scan_num:
            self._assemble_cloud_3d()

    def _assemble_cloud_3d(self) -> None:
        """Assemble accumulated packets into a point cloud."""
        all_points = []
        for pkt in self._point_packets:
            pts = transform_3d(pkt, self._range_min, self._range_max)
            if len(pts) > 0:
                all_points.append(pts)

        if all_points:
            combined = np.concatenate(all_points, axis=0)
        else:
            combined = np.empty((0, 6), dtype=np.float32)

        # Match C++ behavior: subtract scan_period from system timestamp
        scan_period = self._point_packets[0].scan_period if self._point_packets else 0.0
        stamp = (time.time() - scan_period) if self._use_system_timestamp else (
            self._point_packets[0].info.stamp if self._point_packets else 0.0
        )

        self._cloud = PointCloudUnitree(
            stamp=stamp,
            id=self._point_packets[0].info.seq if self._point_packets else 0,
            ring_num=1,
            points=combined,
        )
        self._cloud_ready = True
        self._point_packets.clear()

    def _handle_point_2d(self, data: PointData2D) -> None:
        """Handle 2D scan packet."""
        if data.dirty_index > 0:
            self._dirty_percentage = data.dirty_index

        pts = transform_2d(data, self._range_min, self._range_max)

        stamp = time.time() if self._use_system_timestamp else data.info.stamp

        self._cloud = PointCloudUnitree(
            stamp=stamp,
            id=data.info.seq,
            ring_num=1,
            points=pts,
        )
        self._cloud_ready = True

    def _handle_imu(self, data: ImuDataPacket) -> None:
        """Convert IMU packet to ImuData."""
        self._imu = ImuData(
            stamp=data.info.stamp,
            seq=data.info.seq,
            quaternion=data.quaternion,
            angular_velocity=data.angular_velocity,
            linear_acceleration=data.linear_acceleration,
        )
        self._imu_ready = True

    def _handle_version(self, data: VersionDataPacket) -> None:
        """Store version info."""
        self._version = VersionInfo(
            hardware=data.hw_version,
            firmware=data.sw_version,
            sdk=SDK_VERSION,
            name=data.name,
            date=data.date,
        )
        self._version_ready = True

    # ── Data accessors ───────────────────────────────────────────────────

    def get_point_cloud(self) -> Optional[PointCloudUnitree]:
        """Get the latest assembled point cloud, or None."""
        if self._cloud_ready:
            self._cloud_ready = False
            return self._cloud
        return None

    def get_imu_data(self) -> Optional[ImuData]:
        """Get the latest IMU data, or None."""
        if self._imu_ready:
            self._imu_ready = False
            return self._imu
        return None

    def get_version_firmware(self) -> Optional[str]:
        if self._version is not None:
            return self._version.firmware
        return None

    def get_version_hardware(self) -> Optional[str]:
        if self._version is not None:
            return self._version.hardware
        return None

    def get_version_sdk(self) -> str:
        return SDK_VERSION

    def get_time_delay(self) -> Optional[float]:
        return self._time_delay

    def get_dirty_percentage(self) -> Optional[float]:
        return self._dirty_percentage

    # ── Commands ─────────────────────────────────────────────────────────

    def _send(self, data: bytes) -> None:
        if self._transport is not None:
            self._transport.write(data)

    def start_rotation(self) -> None:
        """Start LiDAR rotation."""
        self._send(build_user_cmd_packet(USER_CMD_STANDBY, 0))

    def stop_rotation(self) -> None:
        """Stop LiDAR rotation."""
        self._send(build_user_cmd_packet(USER_CMD_STANDBY, 1))

    def reset(self) -> None:
        """Reset LiDAR hardware."""
        self._send(build_user_cmd_packet(USER_CMD_RESET, 0))

    def set_work_mode(self, mode: int) -> None:
        """Set LiDAR work mode (see protocol docs for bit definitions)."""
        self._send(build_work_mode_packet(mode))

    def request_version(self) -> None:
        """Request version info from LiDAR."""
        self._send(build_user_cmd_packet(USER_CMD_VERSION_GET, 0))

    def request_latency(self) -> None:
        """Request latency measurement."""
        self._send(build_user_cmd_packet(USER_CMD_LATENCY, 0))

    def sync_timestamp(self) -> None:
        """Sync LiDAR timestamp with system time."""
        now = time.time()
        sec = int(now)
        nsec = int((now - sec) * 1e9)
        self._send(build_timestamp_packet(sec, nsec))

    def set_ip_config(
        self,
        lidar_ip: str = "192.168.1.62",
        user_ip: str = "192.168.1.2",
        gateway: str = "192.168.1.1",
        subnet_mask: str = "255.255.255.0",
        lidar_port: int = 6101,
        user_port: int = 6201,
    ) -> None:
        """Set LiDAR IP address configuration."""

        def _ip_to_tuple(ip: str) -> tuple[int, ...]:
            return tuple(int(x) for x in ip.split("."))

        self._send(build_ip_config_packet(
            _ip_to_tuple(lidar_ip),
            _ip_to_tuple(user_ip),
            _ip_to_tuple(gateway),
            _ip_to_tuple(subnet_mask),
            lidar_port,
            user_port,
        ))

    def clear_buffer(self) -> None:
        """Clear the parser buffer."""
        self._parser.clear()

    @property
    def buffer_cached_size(self) -> int:
        return self._parser.buffer_size

    @property
    def is_connected(self) -> bool:
        return self._transport is not None and self._transport.is_open
