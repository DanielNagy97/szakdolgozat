import cv2

from arpt.grid import Grid
from arpt.capture_device import CaptureDevice
from arpt.video import Video
from arpt.view import View
from arpt.frame_diff import FrameDifference
from arpt.canvas import Canvas
from arpt.window import Window
from arpt.heat_map import HeatMap
from arpt.shift import Shift
# NOTE: Probably it is enought to import only the arpt package.


class Controller(object):
    """
    Controller class
    """

    def __init__(self):
        """
        Initialize the controller.
        """
        self._capture = CaptureDevice(0, 640, 360)
        self._video = Video(self._capture)

        # NOTE: It is not necessarily a web camera.
        self.webcam_win = Window("test", cv2.WINDOW_NORMAL, 0, 0)
        self.vector_field_win = Window("vectorField", cv2.WINDOW_NORMAL, 840, 0)
        self.frame_diff_win = Window("frameDiff", cv2.WINDOW_NORMAL, 420, 0)
        self.heat_map_win = Window("HeatMap", cv2.WINDOW_NORMAL, 840, 350)
        self.plot_win = Window("ResultsPlot", cv2.WINDOW_NORMAL, 0, 350)

        self.resize_window(self.plot_win, 576, 331)
        
        self.frame_diff_canvas = Canvas(self.cap.height, self.cap.width, 1)
        self.vector_field_canvas = Canvas(self.cap.height, self.cap.width, 1, 255)
        self.plot_canvas = Canvas(300, 700, 3)

        self.grid = grid(16, self.cap.width, self.cap.height)

        self.frame_diff = FrameDifference()
        self.heat_map = HeatMap()
        self.shift = Shift(200, 150, 100, 100)
        self.view = View()

    def resize_window(self, win, width, height):
        """
        Resize the window.
        :param win:
        :param width:
        :param height:
        """
        # QUEST: Is method is necessary?
        win.resize(width, height)

    def main_loop(self):
        """
        The main event loop
        """
        while True:
            self.video.get_frame(self.cap)
            if self.video.ret:

                self.frame_diff_control()
                self.grid_control()
                self.heat_map_control()
                self.shift_control()
                self.view_control()

                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

