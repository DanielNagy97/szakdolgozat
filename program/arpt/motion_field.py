import cv2
import numpy as np

from arpt import grid


def calc_optical_flow(prev_image, next_image, grid_resolution):
    """
    Calculate the motion vector field from sequential images.
    :param prev_image: previous image
    :param next_image: current image
    :param grid_resolution: resolution of the vector field as (n_rows, n_columns)
    :return:
    """
    prev_points = grid.calc_centers(prev_image, grid_resolution)
    params = {
        'prevImg': prev_image,
        'nextImg': next_image,
        'prevPts': prev_points,
        'nextPts': None,
        'winSize': (50, 50),
        'maxLevel': 2,
        'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    }
    next_points, status, error = cv2.calcOpticalFlowPyrLK(**params)
    motion_vectors = np.empty((grid_resolution[0], grid_resolution[1], 2), dtype=np.float32)
    k = 0
    for i in range(grid_resolution[0]):
        for j in range(grid_resolution[1]):
            motion_vectors[i][j][0] = next_points[k][1] - prev_points[k][1]
            motion_vectors[i][j][1] = next_points[k][0] - prev_points[k][0]
            if np.isnan(motion_vectors[i][j][0]):
                motion_vectors[i][j][0] = 0.0
            if np.isnan(motion_vectors[i][j][1]):
                motion_vectors[i][j][1] = 0.0
            k += 1
    return motion_vectors


def calc_lengths(motion_vectors):
    """
    Calculate the lengths of the motion vectors.
    """
    n_rows, n_columns, _ = motion_vectors.shape
    lengths = np.empty((n_rows, n_columns), dtype=np.float32)
    for i in range(n_rows):
        for j in range(n_columns):
            dy, dx = motion_vectors[i][j][0], motion_vectors[i][j][1]
            lengths[i][j] = np.sqrt(dx * dx + dy * dy)
    return lengths
