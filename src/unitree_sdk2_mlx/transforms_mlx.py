"""MLX-accelerated coordinate transforms for Apple Silicon.

Same math as transforms.py but using mlx.core for GPU acceleration.
Falls back to NumPy if MLX is not available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

if TYPE_CHECKING:
    import mlx.core as mx

from unitree_sdk2_mlx.protocol import PointData3D, PointData2D


def transform_3d_mlx(
    packet: PointData3D,
    range_min: float = 0.0,
    range_max: float = 100.0,
) -> Union[mx.array, np.ndarray]:
    """MLX-accelerated 3D coordinate transform.

    Returns:
        Array of shape (N, 6): [x, y, z, intensity, time, ring]
        Returns mx.array when MLX is available, np.ndarray otherwise.
    """
    if not HAS_MLX:
        from unitree_sdk2_mlx.transforms import transform_3d

        return transform_3d(packet, range_min, range_max)

    n = min(packet.point_num, len(packet.ranges))
    if n <= 0:
        return mx.zeros((0, 6))

    calib = packet.calib

    # Move data to MLX
    ranges_raw = mx.array(packet.ranges[:n].astype(np.float32))
    intensities = mx.array(packet.intensities[:n].astype(np.float32))

    # Precompute calibration trig (scalar, on CPU)
    sin_beta = float(np.sin(calib.beta_angle))
    cos_beta = float(np.cos(calib.beta_angle))
    sin_xi = float(np.sin(calib.xi_angle))
    cos_xi = float(np.cos(calib.xi_angle))

    cos_beta_sin_xi = cos_beta * sin_xi
    sin_beta_cos_xi = sin_beta * cos_xi
    sin_beta_sin_xi = sin_beta * sin_xi
    cos_beta_cos_xi = cos_beta * cos_xi

    # Validity and range
    valid = ranges_raw >= 1
    range_float = calib.range_scale * (ranges_raw + calib.range_bias)
    valid = valid & (range_float >= packet.range_min) & (range_float <= packet.range_max)
    valid = valid & (range_float >= range_min) & (range_float <= range_max)

    # Zero out invalid ranges so invalid points produce (0,0,0)
    range_f = mx.where(valid, range_float, mx.zeros_like(range_float))
    valid_f = valid.astype(mx.float32)

    # Indices for angle computation
    idx = mx.arange(n, dtype=mx.float32)

    # Angles
    alpha = (packet.angle_min + calib.alpha_angle_bias + idx * packet.angle_increment)
    theta = (packet.com_horizontal_angle_start + calib.theta_angle_bias
             + idx * packet.com_horizontal_angle_step)

    sin_alpha = mx.sin(alpha)
    cos_alpha = mx.cos(alpha)
    sin_theta = mx.sin(theta)
    cos_theta = mx.cos(theta)

    # Intermediate — multiply offsets by valid_f so invalid points get exactly 0
    A = (-cos_beta_sin_xi + sin_beta_cos_xi * sin_alpha) * range_f + calib.b_axis_dist * valid_f
    B = cos_alpha * cos_xi * range_f
    C = (sin_beta_sin_xi + cos_beta_cos_xi * sin_alpha) * range_f

    x = cos_theta * A - sin_theta * B
    y = sin_theta * A + cos_theta * B
    z = C + calib.a_axis_dist * valid_f

    time_rel = idx * packet.time_increment
    ring = mx.ones(n)

    # Stack all columns
    points = mx.stack([x, y, z, intensities, time_rel, ring], axis=-1)

    # Compact valid points preserving original order:
    # Build prefix-sum scatter indices so valid points stay in order
    n_valid = int(mx.sum(valid_f).item())
    if n_valid == 0:
        return mx.zeros((0, 6))

    if n_valid == n:
        # All valid — no compaction needed
        mx.eval(points)
        return points

    # Use cumsum of valid mask to compute destination indices, then gather
    # valid_f cumsum gives 1-based destination index for valid points
    dest = mx.cumsum(valid_f).astype(mx.int32) - 1  # 0-based dest for valid
    # Build output by gathering: for each position where valid, place into dest
    output = mx.zeros((n_valid, 6))
    # Scatter via loop-free approach: extract valid indices via numpy roundtrip
    mx.eval(valid)
    valid_np = np.array(valid)
    valid_indices = np.where(valid_np)[0]
    valid_indices_mx = mx.array(valid_indices.astype(np.int32))
    output = points[valid_indices_mx]

    mx.eval(output)
    return output


def transform_3d_batch_mlx(
    packets: list[PointData3D],
    range_min: float = 0.0,
    range_max: float = 100.0,
) -> Union[mx.array, np.ndarray]:
    """Transform and concatenate multiple 3D packets (full scan).

    This is where MLX shines — batching 18+ packets into one large
    vectorized operation on the GPU.

    Returns mx.array when MLX is available, np.ndarray otherwise.
    """
    if not HAS_MLX:
        from unitree_sdk2_mlx.transforms import transform_3d

        results = [transform_3d(p, range_min, range_max) for p in packets]
        return np.concatenate(results, axis=0) if results else np.empty((0, 6), dtype=np.float32)

    results = [transform_3d_mlx(p, range_min, range_max) for p in packets]
    if not results:
        return mx.zeros((0, 6))

    combined = mx.concatenate(results, axis=0)
    mx.eval(combined)
    return combined
