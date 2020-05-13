import cv2
import numpy as np

from arpt import vector as v


class Grid(object):
    """
    Grid class
    """

    def __init__(self, grid_resolution, dimension):
        """
        Initialize the grid.
        :param grid_density: count of equidistant sections
        :param dimension: a tuple of (width, height) in pixels \
            capture device dimension
        """
        self._old_points = self.calc_centers(dimension, grid_resolution)
        self._new_points = np.empty(self._old_points.shape)

        self._grid_step = np.sum(self._old_points[0])

        rows, cols = grid_resolution

        self._old_points_3D = self._old_points.reshape(rows, cols, 2)
        self._avg_vector_lengths = []
        self._global_euclidean_vectors = np.empty((0, 2), dtype=np.float32)
        self.lk_params = {
            "winSize": (50, 50),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS |
                         cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }

    def calc_centers(self, video_dimension, grid_resolution):
        """
        Calculates
        :param video_dimension: tuple of (n_rows, n_columns) of the video dim
        :param grid_resolution: tuple of (n_rows, n_columns) of the grid res
        :return: NumPy array with size (n_rows * n_columns) x 2
                The first column is for X values, the second is for Y values.
        """
        width = video_dimension[0]
        height = video_dimension[1]
        n_grid_points = grid_resolution[0] * grid_resolution[1]
        centers = np.empty((n_grid_points, 2), dtype=np.float32)
        cell_width = width / grid_resolution[1]
        cell_height = height / grid_resolution[0]
        k = 0
        for i in range(grid_resolution[0]):
            for j in range(grid_resolution[1]):
                centers[k][0] = j * cell_width + (cell_width / 2)
                centers[k][1] = i * cell_height + (cell_height / 2)
                k += 1
        return centers

    def calc_optical_flow(self, video):
        """
        Calculate the optical flow.
        """
        self._new_points, status, error = \
            cv2.calcOpticalFlowPyrLK(video.old_gray_frame,
                                     video.gray_frame,
                                     self._old_points,
                                     None,
                                     **self.lk_params)

    def update_new_points_3D(self):
        """
        Updating the new 3D points.
        Reshaping the new points to 3D.
        """
        self._new_points_3D = \
            self._new_points.reshape(self.old_points_3D.shape)

    def update_vector_lengths(self):
        """
        Update the vector lengths.
        Updating the direction vectors.
        """
        self._euclidean_vectors = \
            np.subtract(self._new_points, self._old_points)
        self._vector_lengths = \
            np.sqrt(np.sum(np.power(self._euclidean_vectors, 2),
                           axis=1))

    def calc_global_resultant_vector(self):
        """
        Calculate the global resultant vector.
        """
        self._global_euclidean_vector = np.mean(self._euclidean_vectors,
                                                axis=0)

        average_vector_length = \
            v.get_vector_length(self._global_euclidean_vector)

        self._avg_vector_lengths.append(average_vector_length)
        self._avg_vector_lengths = self._avg_vector_lengths[-30:]

        self._global_euclidean_vectors =\
            np.append(self._global_euclidean_vectors,
                      np.array([self._global_euclidean_vector],
                               dtype=np.float32),
                      axis=0)
        self._global_euclidean_vectors = self._global_euclidean_vectors[-30:]

    @property
    def old_points(self):
        """
        Get the old points of the vector field.
        :return: np ndarray with float values and shape (n, 2), \
            where n is the number of points
        """
        return self._old_points

    @property
    def new_points(self):
        """
        Get the new points of the vector filed.
        :return: np ndarray with float values and shape (n, 2), \
            where n is the number of points
        """
        return self._new_points

    @property
    def old_points_3D(self):
        """
        Get the old points of the vector field, in 3D array.
        :return: np ndarray with float values and shape (columns, rows, 2)
        """
        return self._old_points_3D

    @property
    def new_points_3D(self):
        """
        Get the new points of the vector filed, in 3D array.
        :return: np ndarray with float values and shape (columns, rows, 2)
        """
        return self._new_points_3D

    @property
    def grid_step(self):
        """
        Get the grid step in pixels.
        :return: number
        """
        return self._grid_step

    @property
    def vector_lengths(self):
        """
        Get the lengths of the vector field's vectors.
        :return: np ndarray with float values
        """
        return self._vector_lengths

    @property
    def avg_vector_lengths(self):
        """
        Get the average lengths of the vector field's vectors \
            for the last 30 frames
        :return: np ndarray with float values
        """
        return self._avg_vector_lengths

    @property
    def global_euclidean_vector(self):
        """
        Get the global direction vector from the vector field
        :return: np ndarray with two elements
        """
        return self._global_euclidean_vector

    @property
    def global_euclidean_vectors(self):
        """
        Get the global direction vectors for the last 30 frame
        :return: np ndarray
        """
        return self._global_euclidean_vectors

    @property
    def euclidean_vectors(self):
        """
        Get the direction vectors of the vector field
        :return: np ndarray with float values
        """
        return self._euclidean_vectors
