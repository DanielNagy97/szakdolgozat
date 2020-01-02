import cv2
import numpy as np

from arpt import vector as v


class Grid(object):
    """
    Grid class
    """
    def __init__(self, grid_density, dimension):
        """
        Initialize the grid.
        :param grid_density: count of equidistant sections
        :param dimension: capture device dimension as a tuple of (width, height) in pixels
        """
        capture_width, capture_height = dimension
        self._old_points = np.empty((0, 2), dtype=np.float32)
        
        self._grid_step = int(capture_width / grid_density)
        for i in range(self._grid_step, capture_height, self._grid_step):
            for j in range(self._grid_step, capture_width, self._grid_step):
                self._old_points = np.append(self._old_points,
                                            np.array([[j, i]], dtype=np.float32),
                                            axis=0)

        self._new_points = np.empty(self._old_points.shape)
        
        rows = grid_density-1
        cols = int(len(self._old_points)/rows)

        # QUEST: Where has it used?
        self._old_points_3D = self._old_points.reshape(cols, rows, 2)
        self._avg_vector_lenghts = []
        self.lk_params = dict(  winSize  = (50, 50),
                                maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def calc_optical_flow(self, video):
        """
        Calculate the optical flow.
        """
        self._new_points, status, error = cv2.calcOpticalFlowPyrLK( video.old_gray_frame,
                                                                    video.gray_frame,
                                                                    self._old_points,
                                                                    None,
                                                                    **self.lk_params)

    def update_new_points_3D(self):
        """
        Update the new points.
        """
        self._new_points_3D = self._new_points.reshape(self.old_points_3D.shape)

    def update_vector_lengths(self):
        """
        Update the vector lengths.
        """
        direction_vectors = np.subtract(self._new_points, self._old_points)
        self._vector_lenghts = np.sqrt(np.sum(np.power(direction_vectors, 2), axis=1))

    def calc_global_resultant_vector(self):
        """
        Calculate the global resultant vector.
        """
        vector_sum = np.array(  [self._old_points.sum(axis=0),
                                self._new_points.sum(axis=0)],
                                dtype=np.float32)
                                
        vector_count = len(self._old_points)
        self._global_direction_vector = v.get_direction_vector(vector_sum)
        average_vector_lenght = v.get_vector_lenght(self._global_direction_vector) / vector_count
        self._avg_vector_lenghts.append(average_vector_lenght)
        self._avg_vector_lenghts = self._avg_vector_lenghts[-30:]
    
    @property
    def old_points(self):
        """
        Get the old points of the vector field.
        :return: np ndarray with float values and shape (n, 2), where n is the number of points
        """
        return self._old_points

    @property
    def new_points(self):
        """
        Get the new points of the vector filed.
        :return: np ndarray with float values and shape (n, 2), where n is the number of points
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
    def vector_lenghts(self):
        """
        Get the lenghts of the vector field's vectors.
        :return: np ndarray with float values
        """
        return self._vector_lenghts

    @property
    def avg_vector_lenghts(self):
        """
        Get the average lenghts of the vector field's vectors for the last 30 frames
        :return: np ndarray with float values
        """
        return self._avg_vector_lenghts

    @property
    def global_direction_vector(self):
        """
        Get the global direction vector from the vector field
        :return: np ndarray with two elements
        """
        return self._global_direction_vector