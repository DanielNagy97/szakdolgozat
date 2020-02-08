import cv2
import numpy as np
import pickle
import os
from datetime import datetime


class Grab(object):
    """
    Grab function class
    """
    def __init__(self):
        self.rect_area = 0
        self._center = []
        self._state = ""

        self.loaded_model = \
            pickle.load(open("./src/ML/trained_models/grab_model.sav", 'rb'))

        self.j = 0
        self.time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dirname = "./src/grab_datas/"+self.time

    def create_data(self, heat_map, frame_diff_canv, grid):
        """
        Create data for training
        :param heat_map: Heat Map object
        :param frame_diff_canv: The frame differencing canvas
        :param grid: The grid object
        """
        if len(heat_map.bounding_rects) == 1:
            x, y, w, h = heat_map.bounding_rects[0]
            if w*h <= 25:
                self.rect_area = w*h

                center = np.array((y + h/2, x + w/2),
                                  dtype=np.uint8)
                center = np.where(center < 2, 2, center)
                max_y, max_x, _ = grid.old_points_3D.shape
                if center[0] > max_y-3:
                    center[0] = max_y-3
                if center[1] > max_x-3:
                    center[1] = max_x-3
                self._center = center
                self.y, self.x = np.subtract(self._center, 2)
                self.w = 5
                self.h = 5

        if self.rect_area != 0 and not heat_map.bounding_rects.any():
            new_canvas = frame_diff_canv.canvas.copy()
            pt1 = tuple(np.uint32(grid.old_points_3D[self.y, self.x]))
            pt2 = tuple(np.uint32(grid.old_points_3D[self.y+self.h-1,
                                                     self.x+self.w-1]))
            # NOTE: The mean of white pixel locations would be better
            self._center_point = \
                tuple(np.uint32(grid.old_points_3D[self._center[0],
                                                   self._center[1]]))
            image = new_canvas[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            image = cv2.resize(image, (16, 16),
                               interpolation=cv2.INTER_AREA)
            # cv2.circle(frame_diff_canv.canvas, self._center_point, 5, 255, 3)
            # cv2.imshow("Grab image", frame_diff_canv.canvas)

            local_direction_vectors = \
                np.subtract(grid.new_points_3D[self.y:self.y+self.h,
                                               self.x:self.x+self.w],
                            grid.old_points_3D[self.y:self.y+self.h,
                                               self.x:self.x+self.w])

            # First 256 feature is the 16*16 image flattened
            # Second 50 feature is the direction vectors (25 pair)
            self.data = \
                np.concatenate((image.flatten(),
                                local_direction_vectors.flatten()),
                               axis=None)

    def save_data(self, heat_map):
        """
        Saving the data to file
        """
        if self.rect_area != 0 and not heat_map.bounding_rects.any():
            if self.j == 0:
                os.makedirs(self.dirname)
            pickle.dump(self.data,
                        open("./"+self.dirname+"/grab"+str(self.j)+".pkl",
                             "wb"))
            print("grab", self.j)
            self.j += 1
            self.rect_area = 0

    def predict(self, heat_map):
        """
        Make prediction for the data
        """
        self._state = ""
        if self.rect_area != 0 and not heat_map.bounding_rects.any():
            X = self.data.reshape(1, -1)
            score = self.loaded_model.predict(X)
            self._state = score[0]
            # print(score)
            self.rect_area = 0

    @property
    def center(self):
        return self._center

    @property
    def center_point(self):
        return self._center_point

    @property
    def state(self):
        return self._state
