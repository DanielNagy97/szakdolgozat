import numpy as np
import cv2

from arpt import vector as v

class HeatMap(object):
    """
    Heatmap class
    """

    def calc_heat_map(self, grid, sensitivity = 10):
        """
        Calculate the heatmap from the grid.
        :param grid: the grid object
        :param sensitivity: sensitivity value for displaying motion vector lenghts
        :return: None
        """

        self.heat_values = np.int32(np.multiply(grid.vector_lenghts, sensitivity))
        self.heat_values = np.where(self.heat_values > 255, 255, self.heat_values)
        self.map = np.zeros(len(self.heat_values), dtype=np.uint8)
        self.map = np.dstack((  np.subtract(255, self.heat_values),
                                self.map,
                                self.heat_values))
        heat_map_shape = list(grid.old_points_3D.shape)
        heat_map_shape[-1] = 3
        heat_map_shape = tuple(heat_map_shape)
        self.map = self.map.reshape(heat_map_shape)
        self.map = np.uint8(self.map)

    def get_motion_points(self, grid, min_area = 2):
        """
        Get the motion points.
        :param grid: the grid object
        :param min_area: below this value all detected blobs with smaller area will be ignored
        :return: None
        """
        gray_map = cv2.cvtColor(self.map, cv2.COLOR_BGR2GRAY)
        ret, thresholded_heat = cv2.threshold(gray_map, 40, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        contours, hierarchy = cv2.findContours(thresholded_heat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.bounding_rects = np.empty((0, 4), dtype=np.uint8)
        self.motion_points_direction = np.empty((0, 2), dtype=np.float32)
        self.motion_points_roots = np.empty((0, 2), dtype=np.float32)

        count = 0

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            rect_area = w * h
            if rect_area < min_area:
                continue
            else:
                count += 1

                local_vector_sum = np.array([grid.old_points_3D[y:y+h, x:x+w].sum(axis=0),
                                            grid.new_points_3D[y:y+h, x:x+w].sum(axis=0)],
                                            dtype=np.float32).sum(axis=1)

                local_vector_sum = np.divide(local_vector_sum, rect_area)

                local_direction_vector = v.get_direction_vector(local_vector_sum)
                local_normalized_direction_vector = v.get_normalized_vector(local_direction_vector)
                
                self.motion_points_roots = np.append(self.motion_points_roots,np.array([local_vector_sum[0]]),axis=0)


                self.motion_points_direction = np.append(   self.motion_points_direction,
                                                            np.array([local_normalized_direction_vector]),
                                                            axis=0)

                self.bounding_rects = np.append(self.bounding_rects,
                                                np.array([(x, y, w, h)],dtype=np.uint8),
                                                axis=0)

    def analyse_two_largest_points(self):
        """
        Find the two largest points.
        :return: None
        NOTE: For now this method is working for two points only...
        """
        self.different_direction = 0.0
        if len(self.motion_points_direction) == 2:
            #normal_vectors = np.multiply(np.flip(self.motion_points_direction,axis=1), [1,-1])
            motion_points_sum = np.abs(np.sum(self.motion_points_direction, axis=0))
            sum_of_a_sum = np.sum(motion_points_sum)

            epsilon = 1

            self.different_direction = (epsilon - sum_of_a_sum) * 100

            if self.different_direction < 0:
                self.different_direction = 0

