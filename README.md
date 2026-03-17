# unitree_sdk2_mlx

**Python/MLX port of the Unitree L2 LiDAR SDK for macOS Apple Silicon**

A pure Python reimplementation of the [Unitree UniLiDAR SDK2](https://github.com/unitreerobotics/unilidar_sdk2) that runs natively on macOS with optional MLX acceleration for point cloud processing on Apple Silicon GPUs.

## Why?

The official Unitree L2 SDK ships prebuilt C++ static libraries for **Linux only** (aarch64 and x86_64). This port:

- **Runs on macOS** — no Linux VM or container needed for development
- **Pure Python** — no C++ compilation, no closed-source binaries
- **MLX-accelerated** — vectorized coordinate transforms on Apple Silicon GPU/ANE
- **Full protocol parity** — reimplements the binary protocol from the open C++ headers
- **Same API shape** — familiar interface for users of the C++ SDK

## Architecture

```
┌─────────────────────────────────────────────────┐
│                UnitreeLidarReader                │  ← High-level API
├─────────────┬───────────────┬───────────────────┤
│  Transport  │    Parser     │    Transforms     │
│  UDP/Serial │  State Machine│  NumPy / MLX      │
├─────────────┴───────────────┴───────────────────┤
│              Protocol (struct-based)             │  ← Binary packet defs
└─────────────────────────────────────────────────┘
```

| Layer | Module | Description |
|-------|--------|-------------|
| Protocol | `protocol.py` | Binary packet structs from `unitree_lidar_protocol.h` |
| CRC | `crc.py` | CRC32 verification (zlib-compatible) |
| Parser | `parser.py` | Frame state machine (reimplements closed-source `libunilidar_sdk2.a`) |
| Transport | `transport/udp.py` | UDP socket (cross-platform) |
| Transport | `transport/serial.py` | Serial with macOS auto-detection (`/dev/cu.usbmodem*`) |
| Transforms | `transforms.py` | NumPy vectorized 3D/2D coordinate transforms |
| Transforms | `transforms_mlx.py` | MLX GPU-accelerated transforms (Apple Silicon) |
| Reader | `reader.py` | High-level `UnitreeLidarReader` (matches C++ API) |
| Types | `types.py` | `PointCloudUnitree`, `ImuData`, `VersionInfo` |

## Installation

```bash
# Clone
git clone https://github.com/RobotFlow-Labs/unitree_sdk2_mlx.git
cd unitree_sdk2_mlx

# Install with uv (recommended)
uv venv .venv --python 3.12
uv pip install -e ".[dev]"

# With MLX acceleration
uv pip install -e ".[mlx,dev]"
```

## Quick Start

```python
from unitree_sdk2_mlx import UnitreeLidarReader
from unitree_sdk2_mlx.protocol import PACKET_POINT_3D, PACKET_IMU

reader = UnitreeLidarReader()
reader.initialize_udp()  # or reader.initialize_serial()
reader.start_rotation()

while True:
    ptype = reader.run_parse()

    if ptype == PACKET_POINT_3D:
        cloud = reader.get_point_cloud()
        if cloud is not None:
            print(f"Cloud: {cloud.size} points")
            # cloud.points is Nx6 numpy array [x, y, z, intensity, time, ring]

    elif ptype == PACKET_IMU:
        imu = reader.get_imu_data()
        if imu is not None:
            print(f"IMU: quat={imu.quaternion}")
```

## Network Setup

Connect the Unitree L2 LiDAR via Ethernet and configure your network interface:

| Parameter | Default |
|-----------|---------|
| Host IP | `192.168.1.2` |
| LiDAR IP | `192.168.1.62` |
| Host Port | `6201` |
| LiDAR Port | `6101` |

## Protocol

The LiDAR communicates via a binary protocol with framed packets:

```
┌──────────┬─────────┬──────────┐
│  Header  │ Payload │   Tail   │
│ 12 bytes │ varies  │ 12 bytes │
└──────────┴─────────┴──────────┘

Header: 0x55 0xAA 0x05 0x0A | packet_type (u32) | packet_size (u32)
Tail:   crc32 (u32) | msg_type_check (u32) | reserve (2B) | 0x00 0xFF
```

| Packet Type | ID | Points/Packet | Description |
|-------------|-----|--------------|-------------|
| 3D Point Cloud | 102 | 300 | xyz + intensity, 18 packets = full scan |
| 2D Laser Scan | 103 | 1800 | Single-ring scan |
| IMU | 104 | — | Quaternion + gyro + accelerometer |
| Version | 105 | — | Hardware/firmware version |

## Tests

```bash
.venv/bin/pytest tests/ -q
```

## Project Structure

```
unitree_sdk2_mlx/
├── src/unitree_sdk2_mlx/
│   ├── __init__.py          # Public API
│   ├── crc.py               # CRC32
│   ├── parser.py            # Frame parser state machine
│   ├── protocol.py          # Binary packet definitions
│   ├── reader.py            # UnitreeLidarReader (main API)
│   ├── transforms.py        # NumPy coordinate transforms
│   ├── transforms_mlx.py    # MLX GPU transforms
│   ├── types.py             # Data types
│   └── transport/
│       ├── base.py          # Transport interface
│       ├── udp.py           # UDP socket
│       └── serial.py        # Serial (macOS-adapted)
├── tests/
├── examples/
│   └── read_udp.py          # UDP example (port of C++ example)
└── pyproject.toml
```

## Credits

- Original SDK: [Unitree Robotics](https://github.com/unitreerobotics/unilidar_sdk2)
- Port by: [AIFLOW LABS / RobotFlowLabs](https://github.com/RobotFlow-Labs)
- Built for the [ANIMA perception stack](https://robotflowlabs.com)

## License

MIT
