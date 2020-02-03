import cv2
import pickle
# from datetime import datetime
# import os


class Ocr_gesture(object):
    """
    OCR-Gesture Recognition class
    """
    def __init__(self):
        self.loaded_model = \
            pickle.load(open("./src/ML/trained_models/ocr_model.sav", 'rb'))
        # self.j = 0
        # self.time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.dirname = "./src/ocr_datas/"+self.time
        # os.makedirs(self.dirname)

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

        # if len(heat_map.motion_points_roots) > 20:
        #     cv2.imwrite("./"+self.dirname+"/im"+str(self.j)+".png",
        #                 canvas.canvas)
        #     self.j += 1

    def predict_motion(self, canvas, heat_map):
        """
        Predicting gesture with trained model
        :param canvas: Visualized gesture
        :param heat_map: The heat-map object
        """
        if len(heat_map.motion_points_roots) > 20:
            score = self.loaded_model.predict([canvas.canvas.flatten()])
            print(score)
