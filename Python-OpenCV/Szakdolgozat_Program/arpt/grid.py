import cv2
import numpy as np

from arpt.vector import Vector as vector


class Grid(object):
    """
    Grid class
    """
    
    def __init__(self, grid_density, capture_width, capture_height):
        """
        Initialize the grid.
        :param grid_density: count of equidistant sections
        :param capture_width: width of the capture device in pixels
        :param capture_height: height of the capture device in pixels
        """
        self._old_points = np.empty((0, 2), dtype=np.float32)
        
        self.grid_step = int(capture_width / grid_density)
        for i in range(self.grid_step, capture_height, self.grid_step):
            for j in range(self.grid_step, capture_width, self.grid_step):
                self._old_points = np.append(self._old_points,
                                            np.array([[j, i]], dtype=np.float32),
                                            axis=0)

        self.new_points = np.empty(self._old_points.shape)
        
        rows = grid_density-1
        cols = int(len(self._old_points)/rows)

        # QUEST: Where has it used?
        self.old_points_3D = self._old_points.reshape(cols, rows, 2)
        self.avg_vector_lenghts = []
        self.lk_params = dict(  winSize  = (50, 50),
                                maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def calc_optical_flow(self, video):
        """
        Calculate the optical flow.
        """
        self.new_points, status, error = cv2.calcOpticalFlowPyrLK(  video.old_gray_frame,
                                                                    video.gray_frame,
                                                                    self._old_points,
                                                                    None,
                                                                    **self.lk_params)

    def update_new_points_3D(self):
        """
        Update the new points.
        """
        self.new_points_3D = self.new_points.reshape(self.old_points_3D.shape)

    def update_vector_lengths(self):
        """
        Update the vector lengths.
        """
        direction_vectors = np.subtract(self.new_points, self._old_points)
        self.vector_lenghts = np.sqrt(np.sum(np.power(direction_vectors, 2), axis=1))

    def calc_global_resultant_vector(self):
        """
        Calculate the global resultant vector.
        """
        vector_sum = vector(np.array(   [self._old_points.sum(axis=0),
                                        self.new_points.sum(axis=0)],
                                        dtype=np.float32))
                                
        vector_count = len(self._old_points)
        self.global_direction_vector = vector_sum.dir_vector()
        average_vector_lenght = self.global_direction_vector.lenght() / vector_count
        self.avg_vector_lenghts.append(average_vector_lenght)
        self.avg_vector_lenghts = self.avg_vector_lenghts[-30:]

