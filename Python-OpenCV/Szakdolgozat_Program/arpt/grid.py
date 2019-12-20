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
        self.old_points_3D = self.old_points.reshape(8,15,2)


        self.lk_params = dict(  winSize  = (50,50),
                                maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def calc_optical_flow(self, video):
        self.new_points, status, error = cv2.calcOpticalFlowPyrLK(video._old_gray_frame, video._gray_frame, self.old_points, None, **self.lk_params)

    def update_new_points_3D(self):
        self.new_points_3D = self.new_points.reshape(8,15,2)

    def update_vector_lenghts(self):
        self.vector_lenghts = np.sqrt(np.sum(np.power(np.subtract(self.new_points,self.old_points),2),axis=1))