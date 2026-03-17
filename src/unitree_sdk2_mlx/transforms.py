"""Vectorized coordinate transforms — NumPy backend.

Ports parseFromPacketToPointCloud() and parseFromPacketPointCloud2D()
from unitree_lidar_utilities.h.
"""

from __future__ import annotations

import numpy as np

from unitree_sdk2_mlx.protocol import PointData3D, PointData2D, CalibParam


def transform_3d(
    packet: PointData3D,
    range_min: float = 0.0,
    range_max: float = 100.0,
) -> np.ndarray:
    """Transform a 3D point data packet to XYZ coordinates.

    Returns:
        Nx6 float32 array: [x, y, z, intensity, time, ring]
        Empty (0,6) array if no valid points.
    """
    n = min(packet.point_num, len(packet.ranges))
    if n <= 0:
        return np.empty((0, 6), dtype=np.float32)

    calib = packet.calib
    ranges_raw = packet.ranges[:n].astype(np.float32)
    intensities = packet.intensities[:n].astype(np.float32)

    # Precompute calibration trig
    sin_beta = np.float32(np.sin(calib.beta_angle))
    cos_beta = np.float32(np.cos(calib.beta_angle))
    sin_xi = np.float32(np.sin(calib.xi_angle))
    cos_xi = np.float32(np.cos(calib.xi_angle))

    cos_beta_sin_xi = cos_beta * sin_xi
    sin_beta_cos_xi = sin_beta * cos_xi
    sin_beta_sin_xi = sin_beta * sin_xi
    cos_beta_cos_xi = cos_beta * cos_xi

    # Validity mask
    valid = ranges_raw >= 1  # jump invalid points (ranges[j] < 1)

    # Range in meters
    range_float = calib.range_scale * (ranges_raw + calib.range_bias)

    # Range limits (both packet and user)
    valid &= (range_float >= packet.range_min) & (range_float <= packet.range_max)
    valid &= (range_float >= range_min) & (range_float <= range_max)

    if not np.any(valid):
        return np.empty((0, 6), dtype=np.float32)

    # Filter to valid points
    range_f = range_float[valid]
    intens = intensities[valid]
    indices = np.where(valid)[0].astype(np.float32)

    # Angles
    alpha = (packet.angle_min + calib.alpha_angle_bias
             + indices * packet.angle_increment)
    theta = (packet.com_horizontal_angle_start + calib.theta_angle_bias
             + indices * packet.com_horizontal_angle_step)

    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Intermediate
    A = (-cos_beta_sin_xi + sin_beta_cos_xi * sin_alpha) * range_f + calib.b_axis_dist
    B = cos_alpha * cos_xi * range_f
    C = (sin_beta_sin_xi + cos_beta_cos_xi * sin_alpha) * range_f

    # Final XYZ
    x = cos_theta * A - sin_theta * B
    y = sin_theta * A + cos_theta * B
    z = C + calib.a_axis_dist

    # Time relative
    time_rel = indices * packet.time_increment

    # Assemble Nx6 [x, y, z, intensity, time, ring]
    n_valid = len(range_f)
    result = np.empty((n_valid, 6), dtype=np.float32)
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = z
    result[:, 3] = intens
    result[:, 4] = time_rel
    result[:, 5] = 1.0  # ring

    return result


def transform_2d(
    packet: PointData2D,
    range_min: float = 0.0,
    range_max: float = 100.0,
) -> np.ndarray:
    """Transform a 2D point data packet to XYZ coordinates.

    Returns:
        Nx6 float32 array: [x, y, z, intensity, time, ring]
    """
    n = min(packet.point_num, len(packet.ranges))
    if n <= 0:
        return np.empty((0, 6), dtype=np.float32)

    calib = packet.calib
    ranges_raw = packet.ranges[:n].astype(np.float32)
    intensities = packet.intensities[:n].astype(np.float32)

    # Validity mask
    valid = ranges_raw >= 1

    # Range in meters
    range_float = calib.range_scale * (ranges_raw + calib.range_bias)

    valid &= (range_float >= packet.range_min) & (range_float <= packet.range_max)
    valid &= (range_float >= range_min) & (range_float <= range_max)

    if not np.any(valid):
        return np.empty((0, 6), dtype=np.float32)

    range_f = range_float[valid]
    intens = intensities[valid]
    indices = np.where(valid)[0].astype(np.float32)

    alpha = packet.angle_min + calib.alpha_angle_bias + indices * packet.angle_increment

    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)

    x = np.zeros_like(range_f)
    y = cos_alpha * range_f
    z = sin_alpha * range_f + calib.a_axis_dist

    time_rel = indices * packet.time_increment

    n_valid = len(range_f)
    result = np.empty((n_valid, 6), dtype=np.float32)
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = z
    result[:, 3] = intens
    result[:, 4] = time_rel
    result[:, 5] = 1.0

    return result
