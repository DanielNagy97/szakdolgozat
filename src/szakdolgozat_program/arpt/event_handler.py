import cv2
import numpy as np


class Event_handler(object):
    """
    Event handler class
    """
    def __init__(self):
        self._up = True
        self._down = False
        self._move = False
        self._position = []
        self.lk_params = \
            dict(winSize=(50, 50),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           10, 0.03))

    def calc_grab_position(self, video):
        """
        Calculating the new position of the point with optical-flow
        NOTE: This one not belongs here
        """
        new_pos, status, error = \
            cv2.calcOpticalFlowPyrLK(video.old_gray_frame,
                                     video.gray_frame,
                                     self._position,
                                     None,
                                     **self.lk_params)
        self._position = new_pos

    def grab(self, grab):
        """
        The Grab event handler
        """
        if grab.state == "grab":
            self.change_state()
            x, y = grab.center_point
            self._position = np.array([[x, y]], dtype=np.float32)

    def ocr_gesture(self, ocr_gesture):
        """
        The OCR-Gesture handler
        """
        if ocr_gesture.state != "":
            # print(ocr_gesture.state)
            pass

    def change_state(self):
        """
        Changing states
        """
        self._up = not self._up
        self._down = not self._down

    @property
    def up(self):
        return self._up

    @property
    def down(self):
        return self._down

    @property
    def move(self):
        return self._move

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_position):
        self._down = new_position
