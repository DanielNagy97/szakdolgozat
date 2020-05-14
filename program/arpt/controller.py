import cv2
import os

from arpt.grid import Grid
from arpt.video import Video
from arpt.view import View
from arpt.frame_difference import FrameDifference
from arpt.window import Window
from arpt.heat_map import HeatMap
from arpt.rotation import Rotation
from arpt.composition import Composition
from arpt.symbol import Symbol
from arpt.blink import Blink
from arpt.dataparser import DataParser

import importlib.util


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
        module_path = os.path.join(source_path, "actions.py")
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

        self.grid = Grid(tuple(stgs['grid']['grid_dimension']),
                         self._video.dimension)
        self.frame_diff = FrameDifference(self._video)

        self.heat_map = HeatMap(self.grid,
                                stgs['heatmap']['sensitivity'],
                                stgs['heatmap']['min_area'])
        self._rotation = Rotation()

        self._symbol = Symbol(self.grid.old_points_3D.shape)
        self._blink = Blink()

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

    def rotation_control(self):
        """
        Controlling the swirl function.
        """
        self._rotation.calc_rotation_points(self.grid,
                                            self.heat_map.bounding_rects,
                                            self._video)

    def symbol_control(self):
        """
        Controlling the motion gesture recognition
        """
        self._symbol.draw_gesture(self.heat_map, self._symbol.canvas)
        self._symbol.predict_motion(self._symbol.canvas, self.heat_map)
        # self._symbol.save_data(self._symbol.canvas, self.heat_map)

    def blink_control(self):
        """
        Controlling the grab function
        """
        self._blink.create_data(self.heat_map,
                                self.frame_diff.canvas,
                                self.grid)
        self._blink.predict(self.heat_map)
        # self._blink.save_data(self.heat_map)
        if self._blink.grabbed:
            self._blink.calc_drag_position(self._video)
            x, y = self._blink.position.ravel()
            cv2.circle(self._video.frame, (int(x), int(y)),
                       3, (0, 0, 255), 4)

    def shiftable_control(self, shiftable_widget):
        """
        Controlling the shiftable function.
        """
        shiftable_widget.calc_shiftable(self.grid, self._video.dimension)

    def expandable_control(self, expandable_widget):
        """
        Controlling the expandable function.
        """
        expandable_widget.calc_expandable(self.grid, self._video.dimension)

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
        grabbable_widget.update_position(self._blink, self._video)

    def tuner_control(self, tuner_widget):
        """
        Controlling the tuner widget.
        """
        tuner_widget.update_value(self._rotation)
        tuner_widget.rotate_widget()

        # Actions from project file codes
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
            if type(widget).__name__ == "Shiftable":
                self.shiftable_control(widget)

            elif type(widget).__name__ == "Expandable":
                self.expandable_control(widget)

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
                    self.view.show_vector_field(self.grid, self._rotation,
                                                self._canvasses['vectorfield'])

                    self.view.show_canvas(self._windows['vectorfield'],
                                          self._canvasses['vectorfield'])

                elif window == 'resultplot':
                    self.view.show_result_plot(self.grid,
                                               self._canvasses['resultplot'])

                    self.view.show_canvas(self._windows['resultplot'],
                                          self._canvasses['resultplot'])
                elif window == 'symbol':
                    self.view.show_canvas(self._windows['symbol'],
                                          self._symbol.canvas)
                elif window == 'symbol-pred':
                    self.view.show_image(self._windows['symbol-pred'],
                                         self._symbol.predicted_gest)
                elif window == 'blink-im':
                    self.view.show_image(self._windows['blink-im'],
                                         self._blink.blink_image)
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
                self.rotation_control()
                self.symbol_control()
                self.blink_control()

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
