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
        self.center = []
        self.image = []
        self.local_direction_vectors = []

        self.j = 0
        self.time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dirname = "./src/grab_datas/"+self.time
        os.makedirs(self.dirname)
        self.loaded_model = \
            pickle.load(open("./src/ML/trained_models/grab_model.sav", 'rb'))

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

                self.center = np.array((y + h/2, x + w/2),
                                       dtype=np.uint8)
                self.center = np.where(self.center < 2, 2, self.center)
                max_y, max_x, _ = grid.old_points_3D.shape
                if self.center[0] > max_y-3:
                    self.center[0] = max_y-3
                if self.center[1] > max_x-3:
                    self.center[1] = max_x-3
                self.y, self.x = np.subtract(self.center, 2)
                self.w = 5
                self.h = 5

        if self.rect_area != 0 and not heat_map.bounding_rects.any():
            new_canvas = frame_diff_canv.canvas.copy()
            pt1 = tuple(np.uint32(grid.old_points_3D[self.y, self.x]))
            pt2 = tuple(np.uint32(grid.old_points_3D[self.y+self.h-1,
                                                     self.x+self.w-1]))
            self.image = new_canvas[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            self.image = cv2.resize(self.image, (16, 16),
                                    interpolation=cv2.INTER_AREA)
            cv2.imshow("Grab image", self.image)

            self.local_direction_vectors = \
                np.subtract(grid.new_points_3D[self.y:self.y+self.h,
                                               self.x:self.x+self.w],
                            grid.old_points_3D[self.y:self.y+self.h,
                                               self.x:self.x+self.w])

            # First 256 feature is the 16*16 image flattened
            # Second 50 feature is the direction vectors (25 pair)
            self.data = \
                np.concatenate((self.image.flatten(),
                                self.local_direction_vectors.flatten()),
                               axis=None)

            # pickle.dump(self.data,
            #             open("./"+self.dirname+"/grab"+str(self.j)+".pkl",
            #                  "wb"))
            # print("jo", self.j)
            # self.j += 1

            X = self.data.reshape(1, -1)
            score = self.loaded_model.predict(X)
            print(score)

            self.rect_area = 0
