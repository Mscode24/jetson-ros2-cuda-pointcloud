# Jetson PointCloud: CUDA-Accelerated ROS 2 RGB-D Processing

![ROS 2](https://img.shields.io/badge/ROS%202-Humble%2FFoxy-blue)
![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Nano-green)
![CUDA](https://img.shields.io/badge/CUDA-Optimized-success)

A high-performance ROS 2 package designed for real-time, filtered RGB PointCloud generation on the **NVIDIA Jetson Nano**. This node optimizes the unprojection of depth maps and RGB packing by offloading computation to the Maxwell GPU via CUDA, ensuring stable performance within strict thermal and power constraints.

## 🚀 Features

- **CUDA-Powered Unprojection:** Uses custom CUDA kernels (via CuPy) to parallelize 3D point generation, significantly reducing CPU load compared to standard unprojection methods.
- **Thermal-Aware Throttling:** Intelligently throttled to ~6 FPS to maintain stable operation on the Jetson Nano without triggering thermal shutdowns.
- **Artifact Filtering:**
    - **Depth Erosion:** Removes "color bleeding" at object edges by eroding the depth mask.
    - **Gradient Filtering:** Eliminates "flying pixels" and depth noise using Sobel-based gradient analysis.
- **Robust Synchronization:** Utilizes `ApproximateTimeSynchronizer` to handle temporal offsets between color and depth streams.
- **Auto-Encoding Detection:** Supports multiple color encodings (BGR8, RGB8, YUV422, Mono8, Bayer) for broad camera compatibility.

## 🛠 Hardware Requirements

- **Processor:** NVIDIA Jetson Nano (or higher).
- **Camera:** Intel RealSense (D435/D455) or any depth camera publishing aligned depth and color images.
- **Memory:** 4GB RAM (recommended).

## 📦 Installation

### 1. Dependencies
Ensure you have ROS 2 and `colcon` installed. You will also need `cupy` configured for your Jetson's CUDA version:

```bash
# Install CuPy (example for Jetson)
pip3 install cupy-cuda102 # Use version matching your CUDA toolkit
```

### 2. Build the Package
Clone this into your ROS 2 workspace `src` folder:

```bash
cd ~/your_ws/src
git clone <your-repo-link>
cd ..
colcon build --packages-select jetson_pointcloud
source install/setup.bash
```

## 🎮 Usage

1. **Launch your camera node** (e.g., `realsense2_camera`). Ensure aligned depth and color streams are active.
2. **Run the PointCloud node**:

```bash
ros2 run jetson_pointcloud pointcloud_node
```

The node will publish to `/camera/custom_pointcloud`. You can visualize this in **Rviz2** using the `camera_link_optical` frame.

## 📊 Performance Notes

On the Jetson Nano, this node achieves:
- **GPU Usage:** ~30-50% during unprojection.
- **Latency:** < 40ms processing time per frame.
- **Stability:** Stable 6 FPS operation at 10W power mode.

## 📜 License
[Your License Choice, e.g., MIT or Apache 2.0]
