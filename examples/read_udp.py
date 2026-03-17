#!/usr/bin/env python3
"""Example: Read point cloud and IMU data via UDP.

Equivalent to example_lidar_udp.cpp from the C++ SDK.

Usage:
    python examples/read_udp.py

Prerequisites:
    - Connect Unitree L2 LiDAR via Ethernet
    - Configure your network interface to 192.168.1.2
"""

import time

from unitree_sdk2_mlx import UnitreeLidarReader
from unitree_sdk2_mlx.protocol import PACKET_POINT_3D, PACKET_POINT_2D, PACKET_IMU


def main():
    reader = UnitreeLidarReader()

    # Initialize UDP connection
    if reader.initialize_udp():
        print("Unilidar initialization failed! Exit here!")
        return
    print("Unilidar initialization succeed!")

    # Start rotation
    reader.start_rotation()
    time.sleep(1)

    # Set work mode (0 = UDP mode)
    work_mode = 0
    print(f"Set LiDAR work mode to: {work_mode}")
    reader.set_work_mode(work_mode)
    time.sleep(1)

    # Reset LiDAR
    reader.reset()
    time.sleep(1)

    # Request version info
    reader.request_version()

    # Wait for version
    for _ in range(100):
        reader.run_parse()
        fw = reader.get_version_firmware()
        if fw is not None:
            hw = reader.get_version_hardware()
            sdk = reader.get_version_sdk()
            print(f"LiDAR hardware version = {hw}")
            print(f"LiDAR firmware version = {fw}")
            print(f"LiDAR SDK version = {sdk}")
            break

    time.sleep(1)

    # Stop and restart rotation
    print("Stop LiDAR rotation ...")
    reader.stop_rotation()
    time.sleep(3)

    print("Start LiDAR rotation ...")
    reader.start_rotation()
    time.sleep(3)

    # Main parse loop
    print("\nStarting main parse loop (Ctrl+C to stop)...")
    try:
        while True:
            result = reader.run_parse()

            if result == PACKET_IMU:
                imu = reader.get_imu_data()
                if imu is not None:
                    print("An IMU msg is parsed!")
                    print(f"\tseq = {imu.seq}, stamp = {imu.stamp:.6f}")
                    print(f"\tquaternion (x,y,z,w) = {imu.quaternion}")
                    print(f"\tangular_velocity = {imu.angular_velocity}")
                    print(f"\tlinear_acceleration = {imu.linear_acceleration}")

            elif result in (PACKET_POINT_3D, PACKET_POINT_2D):
                cloud = reader.get_point_cloud()
                if cloud is not None:
                    print("A Cloud msg is parsed!")
                    print(f"\tstamp = {cloud.stamp:.6f}, id = {cloud.id}")
                    print(f"\tcloud size = {cloud.size}, ringNum = {cloud.ring_num}")
                    if cloud.size > 0:
                        print("\tfirst 10 points (x, y, z, intensity, time, ring):")
                        for i in range(min(10, cloud.size)):
                            p = cloud.points[i]
                            print(f"\t  ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}, "
                                  f"{p[3]:.1f}, {p[4]:.6f}, {int(p[5])})")
                        print("\t  ...")

    except KeyboardInterrupt:
        print("\nStopping...")
        reader.stop_rotation()
        reader.close()


if __name__ == "__main__":
    main()
