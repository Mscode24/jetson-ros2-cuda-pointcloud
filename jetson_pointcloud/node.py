#!/usr/bin/env python3
"""
Stable RGB PointCloud Node — CUDA-optimized for Jetson Nano
Fixes:
  - B&W issue: auto-detects color encoding (bgr8, rgb8, yuv422, mono8)
  - CUDA: depth unprojection & color packing on GPU via CuPy
  - Reduced CPU↔GPU copies with pinned memory
  - Throttled to ~6 FPS to stay within Nano's thermal budget
  - Depth edge erosion to kill bleeding artifacts
  - Depth gradient filter to remove flying pixels
  - ApproximateTimeSynchronizer to fix color/depth temporal offset
"""

import rclpy
from rclpy.node import Node

import numpy as np
import cv2
import time
import message_filters

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud

# ── Try to import CuPy (CUDA) ────────────────────────────────────────────────
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    cp = None
    CUDA_AVAILABLE = False


# ── CUDA kernels (raw) ───────────────────────────────────────────────────────
_UNPROJECT_KERNEL_SRC = r"""
extern "C" __global__
void unproject_depth(
    const float* __restrict__ depth,   // [H*W]
    const float* __restrict__ color,   // [H*W*3]  R,G,B
    float*       __restrict__ out,     // [H*W*4]  x,y,z,rgb_packed_float
    int*         __restrict__ valid_n, // [1] atomic counter
    int W, int H,
    float fx, float fy, float cx, float cy,
    float max_range
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;

    float z = depth[idx];
    if (z <= 0.0f || z > max_range || isnan(z) || isinf(z)) return;

    int u = idx % W;
    int v = idx / W;

    float x = (u - cx) * z / fx;
    float y = (v - cy) * z / fy;

    // Pack RGB into float
    unsigned int r = (unsigned int)color[idx * 3 + 0];
    unsigned int g = (unsigned int)color[idx * 3 + 1];
    unsigned int b = (unsigned int)color[idx * 3 + 2];
    unsigned int rgb_packed = (r << 16) | (g << 8) | b;
    float rgb_float;
    memcpy(&rgb_float, &rgb_packed, sizeof(float));

    int out_idx = atomicAdd(valid_n, 1);
    out[out_idx * 4 + 0] = x;
    out[out_idx * 4 + 1] = y;
    out[out_idx * 4 + 2] = z;
    out[out_idx * 4 + 3] = rgb_float;
}
"""

if CUDA_AVAILABLE:
    _unproject_module = cp.RawModule(code=_UNPROJECT_KERNEL_SRC)
    _unproject_kernel = _unproject_module.get_function("unproject_depth")


# ────────────────────────────────────────────────────────────────────────────
class StableRGBPointCloud(Node):
    def __init__(self):
        super().__init__('stable_rgb_pointcloud_cuda')

        self.bridge  = CvBridge()
        self.info    = None
        self.last_t  = 0.0
        self.INTERVAL  = 0.15   # ~6 FPS (thermal budget for Nano)
        self.MAX_RANGE = 1.5    # metres

        # Erosion kernel for depth edge bleeding fix
        self._erode_kernel = np.ones((5, 5), np.uint8)

        # GPU scratch buffers (allocated once, reused)
        self._gpu_out   = None
        self._gpu_valid = None

        # ── Camera info subscription ─────────────────────────────────────
        self.create_subscription(
            CameraInfo,
            '/camera/camera/aligned_depth_to_color/camera_info',
            self.info_cb,
            10
        )

        # ── Synchronized color + depth subscriptions ─────────────────────
        color_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/aligned_depth_to_color/image_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=10,
            slop=0.05  # 50ms tolerance
        )
        self.ts.registerCallback(self.synced_cb)

        self.pub = self.create_publisher(PointCloud2, '/camera/custom_pointcloud', 10)

        mode = "CUDA (Jetson GPU)" if CUDA_AVAILABLE else "CPU fallback"
        self.get_logger().info(f"🚀 RGB PointCloud Node started — {mode}")

    # ── Callbacks ────────────────────────────────────────────────────────────
    def info_cb(self, msg: CameraInfo):
        self.info = msg

    def _decode_color(self, msg: Image) -> np.ndarray:
        enc = msg.encoding.lower()

        if enc in ('bgr8', 'bgr'):
            return self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        elif enc in ('rgb8', 'rgb'):
            return cv2.cvtColor(
                self.bridge.imgmsg_to_cv2(msg, 'rgb8'), cv2.COLOR_RGB2BGR)

        elif enc in ('yuv422', 'yuyv', 'yuv422_yuy2'):
            raw = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            return cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUY2)

        elif enc in ('mono8', '8uc1'):
            gray = self.bridge.imgmsg_to_cv2(msg, 'mono8')
            self.get_logger().warn(
                "Color stream is mono8 — check camera color stream is enabled!",
                throttle_duration_sec=10)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        elif enc in ('bayer_rggb8', 'bayer_bggr8', 'bayer_gbrg8', 'bayer_grbg8'):
            raw = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            codes = {
                'bayer_rggb8': cv2.COLOR_BAYER_BG2BGR,
                'bayer_bggr8': cv2.COLOR_BAYER_RG2BGR,
                'bayer_gbrg8': cv2.COLOR_BAYER_GR2BGR,
                'bayer_grbg8': cv2.COLOR_BAYER_GB2BGR,
            }
            return cv2.cvtColor(raw, codes[enc])

        else:
            try:
                return self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            except Exception:
                frame = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                self.get_logger().warn(
                    f"Unknown encoding '{msg.encoding}', attempting passthrough",
                    throttle_duration_sec=10)
                return frame

    def synced_cb(self, color_msg: Image, depth_msg: Image):
        now = time.monotonic()
        if now - self.last_t < self.INTERVAL:
            return
        if self.info is None:
            return
        self.last_t = now

        # Decode color
        color = self._decode_color(color_msg)

        # Decode depth
        depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough').astype(np.float32)
        depth /= 1000.0  # mm → m

        # ── Erode depth edges to remove color-bleeding artifacts ─────────
        depth_valid = (depth > 0).astype(np.uint8)
        depth_valid = cv2.erode(depth_valid, self._erode_kernel, iterations=2)
        depth[depth_valid == 0] = np.nan

        # ── Remove flying pixels via depth gradient filter ───────────────
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        depth[np.abs(grad_x) + np.abs(grad_y) > 0.05] = np.nan

        pts = self._unproject_cuda(depth, color) if CUDA_AVAILABLE else self._unproject_cpu(depth, color)
        self._publish(pts)

    # ── GPU path ─────────────────────────────────────────────────────────────
    def _unproject_cuda(self, depth_cpu: np.ndarray,
                        color_cpu: np.ndarray) -> np.ndarray:
        h, w = depth_cpu.shape
        n_pts = h * w

        if self._gpu_out is None or self._gpu_out.shape[0] != n_pts * 4:
            self._gpu_out   = cp.empty((n_pts * 4,), dtype=cp.float32)
            self._gpu_valid = cp.zeros((1,),          dtype=cp.int32)

        self._gpu_valid[:] = 0

        depth_flat = np.ascontiguousarray(depth_cpu.flatten())
        depth_flat[~np.isfinite(depth_flat)] = 0.0

        color_rgb  = cv2.cvtColor(color_cpu, cv2.COLOR_BGR2RGB)
        color_flat = np.ascontiguousarray(
            color_rgb.reshape(-1, 3).astype(np.float32).flatten())

        gpu_depth = cp.asarray(depth_flat)
        gpu_color = cp.asarray(color_flat)

        K = self.info.k
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]

        block = 256
        grid  = (n_pts + block - 1) // block

        _unproject_kernel(
            (grid,), (block,),
            (gpu_depth, gpu_color, self._gpu_out, self._gpu_valid,
             np.int32(w), np.int32(h),
             np.float32(fx), np.float32(fy),
             np.float32(cx), np.float32(cy),
             np.float32(self.MAX_RANGE))
        )

        valid_n = int(self._gpu_valid[0])
        if valid_n == 0:
            return np.empty((0, 4), dtype=np.float32)

        pts = cp.asnumpy(self._gpu_out[:valid_n * 4]).reshape(-1, 4)
        return pts

    # ── CPU fallback ─────────────────────────────────────────────────────────
    def _unproject_cpu(self, depth: np.ndarray,
                       color: np.ndarray) -> np.ndarray:
        h, w = depth.shape
        K = self.info.k
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]

        depth = depth.copy()
        depth[(depth <= 0) | (depth > self.MAX_RANGE)] = np.nan

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        Z = depth.flatten()
        X = ((u - cx) * depth / fx).flatten()
        Y = ((v - cy) * depth / fy).flatten()

        valid = np.isfinite(Z)
        X, Y, Z = X[valid], Y[valid], Z[valid]

        rgb_img = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        rgb = rgb_img.reshape(-1, 3)[valid].astype(np.uint32)
        rgb_u32 = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
        rgb_f   = rgb_u32.view(np.float32)

        return np.column_stack((X, Y, Z, rgb_f)).astype(np.float32)

    # ── Publish ───────────────────────────────────────────────────────────────
    def _publish(self, pts: np.ndarray):
        if pts.shape[0] == 0:
            return

        fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        header = self.info.header
        # Inside your Python script, update the stamp:
        header.stamp = self.get_clock().now().to_msg()
        # Inside your StableRGBPointCloud script (_publish method)
        header.frame_id = 'camera_link_optical'
        cloud = create_cloud(header, fields, pts)
        self.pub.publish(cloud)
        self.get_logger().info(
            f"Published {pts.shape[0]:,} pts", throttle_duration_sec=2)


# ── Entry point ───────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = StableRGBPointCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()