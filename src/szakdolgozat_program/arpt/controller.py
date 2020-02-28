import cv2
from arpt.grid import Grid
from arpt.video import Video
from arpt.view import View
from arpt.frame_difference import FrameDifference
from arpt.canvas import Canvas
from arpt.window import Window
from arpt.heat_map import HeatMap
from arpt.swirl import Swirl
from arpt.composition import Composition
from arpt.ocr_gesture import Ocr_gesture
from arpt.grab import Grab
from arpt.event_handler import Event_handler
from arpt.shift import Shift
from arpt.expand import Expand
from arpt.button import Button
from arpt.grabbable import Grabbable

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
                'resultplot':  Canvas((700, 700), 3),
                'ocr':         Canvas((15, 11), 1)
            }

        self.grid = Grid(16, self._video.dimension)
        self.frame_diff = FrameDifference()
        self.heat_map = HeatMap(self.grid)
        self.swirl = Swirl()

        self._composition = Composition()

        self._ocr = Ocr_gesture()

        self._grab = Grab()

        self._event = Event_handler()

        self.view = View()

        self.scene = [
            {
                'widgets': [
                    Button((40, 350), (100, 100), "./src/button.png"),
                    Grabbable((360, 90), (150, 150), "./src/test.png")
                ]
            },
            {
                'widgets': [
                    Button((510, 40), (100, 100), "./src/button.png"),
                    Shift((60, 40), (120, 156), "./src/test.png"),
                    Expand((10, 10), (200, 300), "./src/expand.png")
                ],
            },
            {
                'widgets': [
                    Expand((400, 150), (200, 300), "./src/expand.png"),
                    Grabbable((140, 50), (150, 150), "./src/test.png")
                ],
            }
        ]

        self.current_scene = 0

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
        if self._grab.grabbed:
            self._grab.calc_grab_position(self._video)
            x, y = self._grab.position.ravel()
            cv2.circle(self._video.frame, (int(x), int(y)),
                       3, (0, 0, 255), 4)

    def event_control(self):
        """
        Controlling the event handler
        """
        self._event.grab(self._grab)
        self._event.ocr_gesture(self._ocr)

        if self._event.grabbed:
            self._event.calc_grab_position(self._video)
            x, y = self._event.position.ravel()
            cv2.circle(self._video.frame, (int(x), int(y)),
                       3, (0, 0, 255), 4)

    def shift_control(self, shift_widget):
        """
        Controlling the shift function.
        """
        shift_widget.calc_shift(self.grid, self._video.dimension, 1.8, 0.8)

    def expand_control(self, expand_widget):
        """
        Controlling the expand function.
        """
        expand_widget.calc_expand(self.grid, self._video.dimension, 1.8, 0.8)

    def button_control(self, button_widget):
        """
        Controlling the button widget.
        """
        button_widget.inspect_button(self.heat_map, self.grid, self._video)

        # For test purposes
        # Pushing the button, will take us to the next "slide"
        if button_widget._pushed:
            self.current_scene += 1

    def grabbable_control(self, grabbable_widget):
        """
        Controlling the grabbable widget.
        """
        grabbable_widget.update_position(self._grab, self._video)

    def update_widgets(self):
        """
        Updating the widgets
        """

        for widget in self.scene[self.current_scene]['widgets']:
            if type(widget).__name__ == "Shift":
                self.shift_control(widget)

            if type(widget).__name__ == "Expand":
                self.expand_control(widget)

            if type(widget).__name__ == "Button":
                self.button_control(widget)

            if type(widget).__name__ == "Grabbable":
                self.grabbable_control(widget)

    def composing_output_video(self):
        """
        Controlling the composition of the video.
        """
        for widget in self.scene[self.current_scene]['widgets']:
            if type(widget).__name__ == "Shift":
                self._composition.draw_widget(widget, self._video)

            if type(widget).__name__ == "Expand":
                self._composition.draw_expand(widget, self._video)

            if type(widget).__name__ == "Widget":
                self._composition.draw_widget(widget, self._video)

            if type(widget).__name__ == "Button":
                self._composition.draw_widget(widget, self._video)

            if type(widget).__name__ == "Grabbable":
                self._composition.draw_widget(widget, self._video)

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
                self.swirl_control()
                self.ocr_gesture_control()
                self.grab_control()

                # self.event_control()

                self.update_widgets()

                self.composing_output_video()
                self.view_control()

                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
                if k == 32:
                    self.current_scene += 1
                    if self.current_scene > len(self.scene)-1:
                        break
            else:
                break

        self._video.release_capture_device()
        cv2.destroyAllWindows()
