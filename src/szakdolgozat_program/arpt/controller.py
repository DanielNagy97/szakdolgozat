import cv2
from arpt.grid import Grid
from arpt.video import Video
from arpt.view import View
from arpt.frame_difference import FrameDifference
from arpt.canvas import Canvas
from arpt.window import Window
from arpt.heat_map import HeatMap
from arpt.swirl import Swirl
from arpt.shift import Shift
from arpt.expand import Expand
from arpt.widget import Widget
from arpt.composition import Composition
from arpt.ocr_gesture import Ocr_gesture
from arpt.grab import Grab
from arpt.event_handler import Event_handler

# NOTE: Probably it is enought to import only the arpt package.
# from arpt import *


class Controller(object):
    """
    Controller class
    """

    def __init__(self, demo=False):
        """
        Initialize the controller.
        """
        self.demo = demo

        self._video = Video(-1, (640, 480), True)

        if self.demo:
            self._windows = {
                'v_stream':      Window("v_stream", cv2.WINDOW_NORMAL, (0, 0))
            }
            cv2.setWindowProperty("v_stream", cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            self._canvasses = {
                'framediff':   Canvas(self._video.dimension, 1),
                'ocr':         Canvas((15, 11), 1)
            }
        else:
            self._windows = {
                'v_stream':    Window("v_stream", cv2.WINDOW_NORMAL,
                                      (0, 0)),
                'vectorfield': Window("vectorField", cv2.WINDOW_NORMAL,
                                      (840, 0)),
                'framediff':   Window("frameDiff", cv2.WINDOW_NORMAL,
                                      (420, 0)),
                'heatmap':     Window("HeatMap", cv2.WINDOW_NORMAL,
                                      (840, 350)),
                'resultplot':  Window("ResultsPlot", cv2.WINDOW_NORMAL,
                                      (0, 350)),
                # 'ocr':         Window("OCR", cv2.WINDOW_NORMAL)
            }
            self._windows['resultplot'].resize(576, 331)

            self._canvasses = {
                'framediff':   Canvas(self._video.dimension, 1),
                'vectorfield': Canvas(self._video.dimension, 1, 255),
                'resultplot':  Canvas((700, 300), 3),
                'ocr':         Canvas((15, 11), 1)
            }

        self.grid = Grid(16, self._video.dimension)
        self.frame_diff = FrameDifference()
        self.heat_map = HeatMap(self.grid)
        self.swirl = Swirl()

        self.shift = Shift((50, 50), (180, 256), "./src/test.png")
        self.expand = Expand((10, 10), (200, 300), "./src/expand.png")

        self.widget = Widget((50, 50), (180, 256), "./src/test.png")

        self._composition = Composition()

        self._ocr = Ocr_gesture()

        self._grab = Grab()

        self._event = Event_handler()

        self.view = View()

    def frame_diff_control(self):
        """
        Controlling the frame differencing function.
        """
        self.frame_diff.apply_frame_difference(self._video,
                                               self._canvasses['framediff'])

    def grid_control(self):
        """
        Controlling vector field grid.
        """
        self.grid.calc_optical_flow(self._video)
        self.grid.update_vector_lengths()
        self.grid.calc_global_resultant_vector()
        self.grid.update_new_points_3D()

    def heat_map_control(self):
        """
        Controlling the motion heat-map function.
        """
        self.heat_map.calc_heat_map(self.grid, 10)
        self.heat_map.get_motion_points(self.grid, 7)
        self.heat_map.analyse_two_largest_points()

    def shift_control(self):
        """
        Controlling the shift function.
        """
        self.shift.calc_shift(self.grid, self._video.dimension, 1.8, 0.8)

    def expand_control(self):
        """
        Controlling the expand function.
        """
        self.expand.calc_expand(self.grid, self._video.dimension, 1.8, 0.8)

    def swirl_control(self):
        """
        Controlling the swirl function.
        """
        self.swirl.calc_swirl(self.grid, self.heat_map.bounding_rects,
                              self._video)

    def ocr_gesture_control(self):
        """
        Controlling the motion gesture recognition
        """
        self._ocr.draw_gesture(self.heat_map, self._canvasses['ocr'])
        self._ocr.predict_motion(self._canvasses['ocr'], self.heat_map)

    def grab_control(self):
        """
        Controlling the grab function
        """
        self._grab.create_data(self.heat_map,
                               self._canvasses['framediff'],
                               self.grid)
        self._grab.predict(self.heat_map)

    def event_control(self):
        """
        Controlling the event handler
        """
        self._event.grab(self._grab)
        self._event.ocr_gesture(self._ocr)

        if self._event.grabbed:
            # print(self._grab.center_point)
            self._event.calc_grab_position(self._video)
            x, y = self._event.position.ravel()
            cv2.circle(self._video.frame, (int(x), int(y)),
                       3, (0, 0, 255), 4)
            # self._event.change_state()

    def composing_output_video(self):
        """
        Controlling the composition of the video.
        """
        self._composition.draw_shift(self.shift, self._video)
        # self._composition.draw_shift(self.widget, self._video)
        self._composition.draw_expand(self.expand, self._video)

    def view_control(self):
        """
        Controlling the View.
        """
        self.view.show_image(self._windows['v_stream'], self._video.frame)
        if not self.demo:
            self.view.show_heat_map(self._windows['heatmap'], self.heat_map)
            self.view.show_canvas(self._windows['framediff'],
                                  self._canvasses['framediff'])
            self.view.show_vector_field(self.grid, self.swirl,
                                        self._windows['vectorfield'],
                                        self._canvasses['vectorfield'])
            self.view.show_global_vector_results(self.grid,
                                                 self._windows['resultplot'],
                                                 self._canvasses['resultplot'])
            # self.view.show_canvas(self._windows['ocr'],
            #                       self._canvasses['ocr'])

    def main_loop(self):
        """
        The main event loop
        """
        while True:
            self._video.get_frame()
            if self._video.ret:

                self.frame_diff_control()
                self.grid_control()
                self.heat_map_control()
                self.shift_control()
                self.expand_control()
                self.swirl_control()
                self.composing_output_video()
                self.ocr_gesture_control()
                self.grab_control()

                self.event_control()

                self.view_control()

                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
            else:
                break

        self._video.release_capture_device()
        cv2.destroyAllWindows()
