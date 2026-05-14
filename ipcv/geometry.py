"""Geometry utilities for camera and localization lessons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics."""

    fx: float
    fy: float
    cx: float
    cy: float

    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Pose2D:
    """A planar pose used by the localization exercises."""

    x: float
    y: float
    theta: float

    def matrix(self) -> np.ndarray:
        c = float(np.cos(self.theta))
        s = float(np.sin(self.theta))
        return np.array(
            [
                [c, -s, self.x],
                [s, c, self.y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def inverse(self) -> "Pose2D":
        transform = np.linalg.inv(self.matrix())
        return pose_from_matrix(transform)

    def compose(self, other: "Pose2D") -> "Pose2D":
        return pose_from_matrix(self.matrix() @ other.matrix())

    def transform_points(self, points_xy: np.ndarray) -> np.ndarray:
        return transform_points_2d(points_xy, self.matrix())


def pose_from_matrix(transform: np.ndarray) -> Pose2D:
    """Convert a 3x3 homogeneous transform to a planar pose."""

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("transform must be 3x3")
    theta = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    return Pose2D(x=float(matrix[0, 2]), y=float(matrix[1, 2]), theta=theta)


def transform_points_2d(points_xy: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous transform to Nx2 points."""

    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape Nx2")
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([points, ones])
    transformed = (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T
    return transformed[:, :2]


def project_points(
    points_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Project Nx3 camera-frame points to Nx2 image coordinates."""

    points = np.asarray(points_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_camera must have shape Nx3")
    z = points[:, 2]
    if np.any(z <= 0):
        raise ValueError("all points must be in front of the camera")
    u = intrinsics.fx * points[:, 0] / z + intrinsics.cx
    v = intrinsics.fy * points[:, 1] / z + intrinsics.cy
    return np.column_stack([u, v])


def backproject_pixels(
    pixels: np.ndarray,
    depth: np.ndarray | float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Backproject image pixels and depth values to camera-frame 3D points."""

    uv = np.asarray(pixels, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("pixels must have shape Nx2")
    z = np.broadcast_to(np.asarray(depth, dtype=np.float64), (uv.shape[0],))
    x = (uv[:, 0] - intrinsics.cx) * z / intrinsics.fx
    y = (uv[:, 1] - intrinsics.cy) * z / intrinsics.fy
    return np.column_stack([x, y, z])


def reprojection_errors(
    expected_pixels: np.ndarray,
    observed_pixels: np.ndarray,
) -> np.ndarray:
    """Return per-point Euclidean reprojection error in pixels."""

    expected = np.asarray(expected_pixels, dtype=np.float64)
    observed = np.asarray(observed_pixels, dtype=np.float64)
    if expected.shape != observed.shape or expected.ndim != 2 or expected.shape[1] != 2:
        raise ValueError("expected and observed pixels must both have shape Nx2")
    return np.linalg.norm(expected - observed, axis=1)
