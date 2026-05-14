import numpy as np

from ipcv.geometry import (
    CameraIntrinsics,
    Pose2D,
    backproject_pixels,
    project_points,
    reprojection_errors,
)


def test_pose_inverse_round_trips_points():
    pose = Pose2D(x=2.0, y=-1.0, theta=0.3)
    points = np.array([[0.0, 0.0], [1.0, 2.0], [-3.0, 4.0]])

    transformed = pose.transform_points(points)
    recovered = pose.inverse().transform_points(transformed)

    np.testing.assert_allclose(recovered, points, atol=1e-9)


def test_project_and_backproject_are_consistent():
    intrinsics = CameraIntrinsics(fx=100.0, fy=120.0, cx=50.0, cy=40.0)
    points = np.array([[1.0, 2.0, 10.0], [-2.0, 1.0, 8.0]])

    pixels = project_points(points, intrinsics)
    recovered = backproject_pixels(pixels, points[:, 2], intrinsics)

    np.testing.assert_allclose(recovered, points, atol=1e-9)


def test_reprojection_errors_reports_pixel_distances():
    expected = np.array([[0.0, 0.0], [3.0, 4.0]])
    observed = np.array([[0.0, 0.0], [0.0, 0.0]])

    errors = reprojection_errors(expected, observed)

    np.testing.assert_allclose(errors, [0.0, 5.0])
