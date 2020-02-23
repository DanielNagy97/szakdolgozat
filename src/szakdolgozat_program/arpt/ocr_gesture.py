import cv2
import pickle
import numpy as np
from datetime import datetime
import os


class Ocr_gesture(object):
    """
    OCR-Gesture Recognition class
    """
    def __init__(self):
        self._state = ""
        self.loaded_model = \
            pickle.load(open("./src/ML/trained_models/ocr_model.sav", 'rb'))
        self.j = 0
        self.time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dirname = "./src/ocr_datas/"+self.time

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
        if (len(heat_map.motion_points_roots) > 20 and not
           heat_map.bounding_rects.any()):
            score = self.loaded_model.predict([canvas.canvas.flatten()])
            self._state = score[0]
            # cv2.imshow("ocr-canvas", canvas.canvas)
            heat_map.motion_points_roots = np.empty((0, 2), dtype=np.uint8)

            # Showing what gesture got predicted
            im = cv2.imread("./src/"+score[0]+".png", 0)
            cv2.imshow("Detected gesture", im)

    def save_data(self, heat_map, canvas):
        if (len(heat_map.motion_points_roots) > 20 and not
           heat_map.bounding_rects.any()):
            if self.j == 0:
                os.makedirs(self.dirname)
            cv2.imwrite("./"+self.dirname+"/im"+str(self.j)+".png",
                        canvas.canvas)
            print("ocr"+str(self.j))
            heat_map.motion_points_roots = np.empty((0, 2), dtype=np.uint8)
            self.j += 1

    def add_new_gesture(self, heat_map, canvas, number_of_gest, repeatance):
        """
        Chanche for user to add new gestures
        Each gesture should be done n times before accepting
        Then it's needs to be verified
        """
        # TODO: Saving user gestures to a structure!!!
        if self.j < repeatance:
            self.save_data(heat_map, canvas)

    def verify_user_gestures(self):
        """
        Verifying user gestures
        Nesting user gestures from same class, then inspecting them
        Calculating distance between user gestures
        different values
        cols, rows
        mean of black pixel location
        distance in percentage
        what to accept
        If the results are poor making advice
        Giving advice how to improve user gestures
        """
        pass

    def generating_training_data_from_user_gestures(self):
        """
        Generating training data from user gestures
        With different methods:
        like noising, shifting, scaling
        """
        pass

    @property
    def state(self):
        return self._state
