import cv2

from arpt.grid import Grid
from arpt.video import Video
from arpt.view import View
from arpt.frame_difference import FrameDifference
from arpt.window import Window
from arpt.heat_map import HeatMap
from arpt.swirl import Swirl
from arpt.composition import Composition
from arpt.ocr_gesture import Ocr_gesture
from arpt.event_handler import Event_handler
from arpt.grab import Grab
from arpt.dataparser import DataParser

import importlib.util

# NOTE: Probably it is enought to import only the arpt package.
# from arpt import *


class Controller(object):
    """
    Controller class
    """
    def __init__(self, source_path, demo=False):
        """
        Initialize the controller.
        :param source_path: The path of the project file
        :demo: When True, only the output will be shown
        """
        module_path = source_path+"actions.py"
        spec = \
            importlib.util.spec_from_file_location("module.name", module_path)
        self.actions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.actions)

        self.demo = demo

        data_parser = DataParser()

        stgs = data_parser.read_json(source_path,
                                     'preferences/settings.json')

        self._video = Video(stgs['video']['source'],
                            tuple(stgs['video']['dimension']),
                            tuple(stgs['video']['resize']),
                            stgs['video']['to_flip'])

        self._transparency = stgs['transparency']

        if self.demo:
            self._windows = {'v_stream': Window("v_stream",
                                                cv2.WINDOW_NORMAL,
                                                (0, 0))}
            cv2.setWindowProperty("v_stream",
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        else:
            self._windows = data_parser.build_windows_from_pref(source_path)
            self._canvasses = \
                data_parser.build_canvasses_from_pref(self._video, source_path)

        self.grid = Grid(stgs['grid']['gridstep'], self._video.dimension)
        self.frame_diff = FrameDifference(self._video)

        self.heat_map = HeatMap(self.grid,
                                stgs['heatmap']['sensitivity'],
                                stgs['heatmap']['min_area'])
        self.swirl = Swirl()

        self._ocr = Ocr_gesture()
        self._grab = Grab()

        self._event = Event_handler()

        self._composition = Composition()
        self.view = View()

        self.scene = data_parser.build_scene_from_project_file(source_path)
        self.current_scene = 0

    def frame_diff_control(self):
        """
        Controlling the frame differencing function.
        """
        self.frame_diff.apply_frame_difference(self._video)

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
        self.heat_map.calc_heat_map(self.grid)
        self.heat_map.get_motion_points(self.grid)
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
        self._ocr.draw_gesture(self.heat_map, self._ocr.canvas)
        self._ocr.predict_motion(self._ocr.canvas, self.heat_map)

    def grab_control(self):
        """
        Controlling the grab function
        """
        self._grab.create_data(self.heat_map,
                               self.frame_diff.canvas,
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
        pass

    def shift_control(self, shift_widget):
        """
        Controlling the shift function.
        """
        shift_widget.calc_shift(self.grid, self._video.dimension)

    def expand_control(self, expand_widget):
        """
        Controlling the expand function.
        """
        expand_widget.calc_expand(self.grid, self._video.dimension)

    def button_control(self, button_widget):
        """
        Controlling the button widget.
        """
        button_widget.inspect_button(self.heat_map, self.grid, self._video)

        # Actions from project file codes
        getattr(self.actions, button_widget.action)(self, button_widget)

    def grabbable_control(self, grabbable_widget):
        """
        Controlling the grabbable widget.
        """
        grabbable_widget.update_position(self._grab, self._video)

    def tuner_control(self, tuner_widget):
        """
        Controlling the grabbable widget.
        """
        tuner_widget.update_value(self.swirl)
        tuner_widget.rotate_widget()

        getattr(self.actions, tuner_widget.action)(self, tuner_widget)

    def rollable_control(self, rollable_widget):
        """
        Controlling the rollable widget.
        """
        rollable_widget.calc_roll(self.grid, self._video.dimension)

    def update_widgets(self):
        """
        Updating the widgets
        """
        for widget in self.scene[self.current_scene]['widgets']:
            if type(widget).__name__ == "Shift":
                self.shift_control(widget)

            elif type(widget).__name__ == "Expand":
                self.expand_control(widget)

            elif type(widget).__name__ == "Button":
                self.button_control(widget)

            elif type(widget).__name__ == "Grabbable":
                self.grabbable_control(widget)

            elif type(widget).__name__ == "Tuner":
                self.tuner_control(widget)

            elif type(widget).__name__ == "Rollable":
                self.rollable_control(widget)

    def composing_output_video(self):
        """
        Controlling the composition of the video.
        """
        for widget in self.scene[self.current_scene]['widgets']:
            self._composition.draw_widget(widget, self._video,
                                          self._transparency)

    def view_control(self):
        """
        Controlling the View.
        """
        if not self.demo:
            for window in self._windows:
                if window == 'v_stream':
                    self.view.show_image(self._windows['v_stream'],
                                         self._video.frame)
                elif window == 'heatmap':
                    self.view.show_heat_map(self._windows['heatmap'],
                                            self.heat_map)
                elif window == 'framediff':
                    self.view.show_canvas(self._windows['framediff'],
                                          self.frame_diff.canvas)
                elif window == 'vectorfield':
                    self.view.show_vector_field(self.grid, self.swirl,
                                                self._windows['vectorfield'],
                                                self._canvasses['vectorfield'])
                elif window == 'resultplot':
                    self.view.show_result_plot(self.grid,
                                               self._windows['resultplot'],
                                               self._canvasses['resultplot'])
                elif window == 'ocr':
                    self.view.show_canvas(self._windows['ocr'],
                                          self._ocr.canvas)
                elif window == 'ocr-pred':
                    self.view.show_image(self._windows['ocr-pred'],
                                         self._ocr.predicted_gest)
                elif window == 'grab-im':
                    self.view.show_image(self._windows['grab-im'],
                                         self._grab.grab_image)
        else:
            self.view.show_image(self._windows['v_stream'],
                                 self._video.frame)

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
