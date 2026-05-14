import numpy as np

from ipcv.image_processing import canny_edges, detect_corners, sobel_edges


def test_sobel_edges_highlights_intensity_step():
    image = np.zeros((8, 8), dtype=float)
    image[:, 4:] = 1.0

    edges = sobel_edges(image)

    assert edges.shape == image.shape
    assert edges[:, 3:5].mean() > edges[:, :2].mean()


def test_canny_edges_returns_boolean_map():
    image = np.zeros((16, 16), dtype=np.uint8)
    image[4:12, 4:12] = 255

    edges = canny_edges(image)

    assert edges.shape == image.shape
    assert edges.dtype == bool
    assert edges.any()


def test_detect_corners_returns_feature_points():
    image = np.zeros((16, 16), dtype=np.uint8)
    image[4:12, 4:12] = 255

    corners = detect_corners(image, max_corners=4)

    assert len(corners) > 0
    assert all(0 <= corner.x < 16 and 0 <= corner.y < 16 for corner in corners)
