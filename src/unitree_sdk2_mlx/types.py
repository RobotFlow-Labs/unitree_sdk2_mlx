"""Data types matching unitree_lidar_utilities.h."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PointUnitree:
    """Single LiDAR point."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    intensity: float = 0.0
    time: float = 0.0  # relative time from cloud stamp
    ring: int = 1


@dataclass
class PointCloudUnitree:
    """Point cloud assembled from multiple packets."""

    stamp: float = 0.0
    id: int = 0
    ring_num: int = 1
    # Nx6 float32 array: [x, y, z, intensity, time, ring]
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 6), dtype=np.float32))

    @property
    def size(self) -> int:
        return len(self.points)


@dataclass
class ImuData:
    """IMU data from the LiDAR."""

    stamp: float = 0.0
    seq: int = 0
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # x, y, z, w
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class VersionInfo:
    """LiDAR hardware/firmware version."""

    hardware: str = ""
    firmware: str = ""
    sdk: str = "0.1.0"
    name: str = ""
    date: str = ""


@dataclass
class InsideState:
    """Internal LiDAR state."""

    sys_rotation_period: int = 0
    com_rotation_period: int = 0
    dirty_index: float = 0.0
    packet_lost_up: float = 0.0
    packet_lost_down: float = 0.0
    apd_temperature: float = 0.0
    apd_voltage: float = 0.0
    laser_voltage: float = 0.0
    imu_temperature: float = 0.0
