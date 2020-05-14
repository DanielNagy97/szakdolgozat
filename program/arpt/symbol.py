import cv2
import pickle
import numpy as np
from datetime import datetime
import os
from arpt.canvas import Canvas


class Symbol(object):
    """
    Symbol Recognition class
    """
    def __init__(self, shape):
        """
        Initalize the Symbol function
        """
        self.shape = tuple([shape[1], shape[0]])
        self._canvas = Canvas(self.shape, 1)
        self._state = ""
        self.loaded_model = \
            pickle.load(open("./src/ML/trained_models/symbol_model.sav", 'rb'))
        self.j = 0
        self.time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dirname = "./src/symbol_datas/"+self.time

        self._predicted_gest = np.uint8(np.empty((11, 15)))

    def draw_gesture(self, heat_map, canvas):
        """
        Drawing the gesture to a canvas
        :param heat_map: The heat-map object
        :param canvas: Visualized gesture
        """
        canvas.fill(255)
        i = 1
        while i < len(heat_map.motion_points_roots):
            cv2.line(canvas.canvas,
                     tuple(heat_map.motion_points_roots[i-1]),
                     tuple(heat_map.motion_points_roots[i]),
                     0,
                     1)
            i += 1

    def predict_motion(self, canvas, heat_map):
        """
        Predicting gesture with trained model
        :param canvas: Visualized gesture
        :param heat_map: The heat-map object
        """
        self._state = ""
        if (len(heat_map.motion_points_roots) > 15 and not
           heat_map.bounding_rects.any()):

            gesture_im = self.preprocess_gesture(canvas,
                                                 heat_map.map_shape[1])

            score = self.loaded_model.predict([gesture_im.flatten()])
            self._state = score[0]

            heat_map.motion_points_roots = np.empty((0, 2), dtype=np.uint8)

            # Showing what gesture got predicted
            im = cv2.imread("./src/"+score[0]+".png", 0)
            self._predicted_gest = im

    def save_data(self, canvas, heat_map):
        """
        Saving data to disk
        :param heat_map: The Heat-map object
        :param canvas: The OCR-canvas
        """
        if (len(heat_map.motion_points_roots) > 15 and not
           heat_map.bounding_rects.any()):
            canvas.canvas = self.preprocess_gesture(canvas,
                                                    heat_map.map_shape[1])
            if self.j == 0:
                os.makedirs(self.dirname)
            cv2.imwrite("./"+self.dirname+"/im_"+str(self.j)+".png",
                        canvas.canvas)
            print("symbol_"+str(self.j))
            heat_map.motion_points_roots = np.empty((0, 2), dtype=np.uint8)
            self.j += 1

    def preprocess_gesture(self, canvas, width):
        """
        Preprocessing
        Useful, when working with different grid aspect ratios
        Making 3D canvas to 2D (?)
        Cropping gesture image from canvas
        Then enlarging it to 15 x 11
        :return: the preprocessed gesture image as np.ndarray
        """
        gesture_im = canvas.canvas.copy()
        gesture_im = gesture_im.reshape((-1, width))

        mask = gesture_im < 1
        coords = np.argwhere(mask)

        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1

        cropped = gesture_im[x0:x1, y0:y1]

        # returned as 15 x 11, because of the estimator is trained like that
        return cv2.resize(cropped, (15, 11), interpolation=cv2.INTER_AREA)

    @property
    def state(self):
        return self._state

    @property
    def canvas(self):
        return self._canvas

    @property
    def predicted_gest(self):
        return self._predicted_gest
