"""Tests for coordinate transforms."""

import math

import numpy as np

from unitree_sdk2_mlx.protocol import PointData3D, PointData2D, CalibParam, DataInfo
from unitree_sdk2_mlx.transforms import transform_3d, transform_2d


def _make_3d_packet(
    n_points: int = 10,
    range_val: int = 1000,  # mm
    calib: CalibParam | None = None,
) -> PointData3D:
    """Create a synthetic 3D point packet for testing."""
    if calib is None:
        calib = CalibParam(
            a_axis_dist=0.0,
            b_axis_dist=0.0,
            theta_angle_bias=0.0,
            alpha_angle_bias=0.0,
            beta_angle=0.0,
            xi_angle=0.0,
            range_bias=0.0,
            range_scale=0.001,  # mm to meters
        )

    ranges = np.full(300, range_val, dtype=np.uint16)
    intensities = np.full(300, 128, dtype=np.uint8)

    return PointData3D(
        info=DataInfo(seq=1, payload_size=0, stamp_sec=0, stamp_nsec=0),
        calib=calib,
        com_horizontal_angle_start=0.0,
        com_horizontal_angle_step=0.01,
        scan_period=0.1,
        range_min=0.0,
        range_max=50.0,
        angle_min=-math.pi / 4,
        angle_increment=math.pi / 150,
        time_increment=0.001,
        point_num=n_points,
        ranges=ranges,
        intensities=intensities,
        dirty_index=0.0,
    )


def _make_2d_packet(
    n_points: int = 10,
    range_val: int = 2000,
) -> PointData2D:
    """Create a synthetic 2D point packet for testing."""
    calib = CalibParam(
        a_axis_dist=0.0,
        b_axis_dist=0.0,
        theta_angle_bias=0.0,
        alpha_angle_bias=0.0,
        beta_angle=0.0,
        xi_angle=0.0,
        range_bias=0.0,
        range_scale=0.001,
    )

    ranges = np.full(1800, range_val, dtype=np.uint16)
    intensities = np.full(1800, 200, dtype=np.uint8)

    return PointData2D(
        info=DataInfo(seq=1, payload_size=0, stamp_sec=0, stamp_nsec=0),
        calib=calib,
        scan_period=0.1,
        range_min=0.0,
        range_max=50.0,
        angle_min=-math.pi / 4,
        angle_increment=math.pi / 900,
        time_increment=0.001,
        point_num=n_points,
        ranges=ranges,
        intensities=intensities,
        dirty_index=0.0,
    )


def test_transform_3d_basic():
    pkt = _make_3d_packet(n_points=10, range_val=1000)
    pts = transform_3d(pkt)
    assert pts.shape == (10, 6)
    assert pts.dtype == np.float32
    # All intensities should be 128
    np.testing.assert_array_equal(pts[:, 3], 128.0)
    # Ring should all be 1
    np.testing.assert_array_equal(pts[:, 5], 1.0)


def test_transform_3d_range_filter():
    pkt = _make_3d_packet(n_points=10, range_val=1000)
    # range_scale=0.001 means range = 0.001 * 1000 = 1.0 meter
    pts = transform_3d(pkt, range_min=2.0, range_max=100.0)
    # 1.0m < 2.0m min, so all filtered
    assert pts.shape[0] == 0


def test_transform_3d_invalid_ranges():
    pkt = _make_3d_packet(n_points=10, range_val=0)
    pts = transform_3d(pkt)
    assert pts.shape[0] == 0  # All invalid (range < 1)


def test_transform_3d_empty():
    pkt = _make_3d_packet(n_points=0)
    pts = transform_3d(pkt)
    assert pts.shape == (0, 6)


def test_transform_2d_basic():
    pkt = _make_2d_packet(n_points=20, range_val=2000)
    pts = transform_2d(pkt)
    assert pts.shape == (20, 6)
    # All x should be 0 for 2D
    np.testing.assert_array_equal(pts[:, 0], 0.0)
    # Intensities should be 200
    np.testing.assert_array_equal(pts[:, 3], 200.0)


def test_transform_3d_with_calibration():
    """Test with non-trivial calibration parameters."""
    calib = CalibParam(
        a_axis_dist=0.05,
        b_axis_dist=0.02,
        theta_angle_bias=0.01,
        alpha_angle_bias=0.005,
        beta_angle=0.1,
        xi_angle=0.2,
        range_bias=0.0,
        range_scale=0.001,
    )
    pkt = _make_3d_packet(n_points=5, range_val=5000, calib=calib)
    pts = transform_3d(pkt)
    assert pts.shape == (5, 6)
    # Points should be non-zero with calibration
    assert np.any(pts[:, 0] != 0)
    assert np.any(pts[:, 1] != 0)
