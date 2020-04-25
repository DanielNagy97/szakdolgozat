import numpy as np


def calc_centers(image, grid_resolution):
    """
    Calculates
    :param image: OpenCV image (for image dimension)
    :param grid_resolution: tuple of (n_rows, n_columns) of the grid resolution
    :return: NumPy array with size (n_rows * n_columns) x 2
             The first column is for X values, the second is for Y values.
    """
    width = image.shape[1]
    height = image.shape[0]
    n_grid_points = grid_resolution[0] * grid_resolution[1]
    centers = np.empty((n_grid_points, 2), dtype=np.float32)
    cell_width = width / grid_resolution[0]
    cell_height = height / grid_resolution[1]
    k = 0
    for i in range(grid_resolution[1]):
        for j in range(grid_resolution[0]):
            centers[k][0] = j * cell_width + (cell_width / 2)
            centers[k][1] = i * cell_height + (cell_height / 2)
            k += 1
    return centers