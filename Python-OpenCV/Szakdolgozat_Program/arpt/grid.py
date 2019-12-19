import numpy as np
import cv2

class grid():
    def __init__(self,grid_density, cap_width, cap_height):
        self.grid_step = int(cap_width/grid_density)
        for i in range(self.grid_step, cap_height, self.grid_step):
            for j in range(self.grid_step, cap_width, self.grid_step):
                if i == self.grid_step and j == self.grid_step:
                    self.old_points = np.array([[j, i]], dtype=np.float32)
                else:
                    self.old_points = np.concatenate((self.old_points, np.array([[j, i]], dtype=np.float32)))
        self.new_points = np.empty(self.old_points.shape)

        self.lk_params = dict(  winSize  = (50,50),
                                maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def calc_optical_flow(self, old_gray_frame, gray_frame):
        self.new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray_frame, gray_frame, self.old_points, None, **self.lk_params)
        