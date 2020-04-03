import numpy as np
import cv2

from arpt import vector as v


class HeatMap(object):
    """
    Heatmap class
    """

    def __init__(self, grid, sensitivity=10, min_area=2):
        """
        Initialize the Heat Map
        :param grid: the grid object
        :param sensitivity: sensitivity value \
            for displaying motion vector lengths
        :param min_area: minimum area of motion blob to analyse
        """
        self._map_shape = list(grid.old_points_3D.shape)
        self._map_shape[-1] = 3
        self._map_shape = tuple(self._map_shape)

        self._motion_points_roots = np.empty((0, 2), dtype=np.uint8)

        self.sensitivity = sensitivity
        self.min_area = min_area

    def calc_heat_map(self, grid):
        """
        Calculate the heatmap from the grid.
        :param grid: the grid object
        :return: None
        """
        heat_values = np.int32(np.multiply(grid.vector_lengths,
                                           self.sensitivity))
        heat_values = np.where(heat_values > 255, 255, heat_values)
        self._map = np.zeros(len(heat_values), dtype=np.uint8)
        self._map = np.dstack((np.subtract(255, heat_values),
                               self._map,
                               heat_values))
        self._map = self._map.reshape(self._map_shape)
        self._map = np.uint8(self._map)

    def get_motion_points(self, grid):
        """
        Get the motion points.
        :param grid: the grid object
        :return: None
        """
        gray_map = cv2.cvtColor(self._map, cv2.COLOR_BGR2GRAY)
        ret, thresholded_heat = cv2.threshold(gray_map, 40, 255,
                                              cv2.ADAPTIVE_THRESH_MEAN_C)
        contours, hierarchy = cv2.findContours(thresholded_heat,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

        self._bounding_rects = np.empty((0, 4), dtype=np.uint8)
        self._motion_points_direction = np.empty((0, 2), dtype=np.float32)

        count = 0

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            rect_area = w * h
            if rect_area >= self.min_area:
                count += 1

                euclidean_vectors = np.reshape(grid.euclidean_vectors,
                                               grid.old_points_3D.shape)
                local_euclidean_vector = \
                    np.mean(euclidean_vectors[y:y+h, x:x+w].reshape((-1, 2)),
                            axis=0)

                local_normalized_euclidean_vector = \
                    v.get_normalized_vector(local_euclidean_vector)

                m_diff_eps = 4
                if len(self.motion_points_roots) == 0:
                    self._motion_points_roots = \
                        np.append(self._motion_points_roots,
                                  np.array([(x + w/2, y + h/2)],
                                           dtype=np.uint8),
                                  axis=0)
                else:
                    x_diff = abs(x+w/2 - self.motion_points_roots[-1][0])
                    y_diff = abs(y+h/2 - self.motion_points_roots[-1][1])
                    if x_diff <= m_diff_eps and y_diff <= m_diff_eps:
                        self._motion_points_roots = \
                            np.append(self._motion_points_roots,
                                      np.array([(x + w/2, y + h/2)],
                                               dtype=np.uint8),
                                      axis=0)

                self._motion_points_direction = \
                    np.append(self._motion_points_direction,
                              np.array([local_normalized_euclidean_vector]),
                              axis=0)

                self._bounding_rects = np.append(self._bounding_rects,
                                                 np.array([(x, y, w, h)],
                                                          dtype=np.uint8),
                                                 axis=0)

        self._motion_points_roots = self._motion_points_roots[-33:]
        if len(self._bounding_rects) == 0:
            self._motion_points_roots = self._motion_points_roots[1:]

    def analyse_two_largest_points(self):
        """
        Find the two largest points.
        :return: None
        NOTE: For now this method is working for two points only...
        """
        self._different_direction = 0.0
        if len(self._motion_points_direction) == 2:
            # normal_vectors = \
            # np.multiply(np.flip(self._motion_points_direction,axis=1),
            #                     [1,-1])
            motion_points_sum = np.abs(np.sum(self._motion_points_direction,
                                              axis=0))
            sum_of_a_sum = np.sum(motion_points_sum)

            epsilon = 1

            self._different_direction = (epsilon - sum_of_a_sum) * 100

            if self._different_direction < 0:
                self._different_direction = 0

    @property
    def map(self):
        """
        Get the map.
        :return: np ndarray with int values from 0-255 \
            representing colors and shape (columns, rows, 3)
        """
        return self._map

    @property
    def map_shape(self):
        """
        Get the shape of the map.
        :return: np ndarray with int values from 0-255 \
            representing colors and shape (columns, rows, 3)
        """
        return self._map_shape

    @property
    def bounding_rects(self):
        """
        Get the bounding rects of the heat map blobs.
        :return: np ndarray with shape (n, 4), \
            each element is representing a rect in (x,y,w,h) format
        """
        return self._bounding_rects

    @property
    def motion_points_direction(self):
        """
        Get the direction of motion points.
        :return: np ndarray with shape (n, 2), \
            each element is representing a direction vector
        """
        return self._motion_points_direction

    @property
    def motion_points_roots(self):
        """
        Get the roots of motion points
        :return: np ndarray with shape (n, 2)
        """
        return self._motion_points_roots

    @motion_points_roots.setter
    def motion_points_roots(self, new):
        """
        Get the roots of motion points
        :return: np ndarray with shape (n, 2)
        """
        self._motion_points_roots = new

    @property
    def different_direction(self):
        """
        Get the difference of the two largest motion point's direction \
            in percentage.
        :return: float
        """
        return self._different_direction
