import cv2
from arpt import *
# NOTE: Probably it is enought to import only the arpt package.

class Controller(object):
    """
    Controller class
    """

    def __init__(self):
        """
        Initialize the controller.
        """
        self._capture = CaptureDevice(0, 0, 0)
        self._video = Video(self._capture)

        # NOTE: It is not necessarily a web camera.
        self.webcam_win = Window("test", cv2.WINDOW_NORMAL, (0, 0))
        self.vector_field_win = Window("vectorField", cv2.WINDOW_NORMAL, (840, 0))
        self.frame_diff_win = Window("frameDiff", cv2.WINDOW_NORMAL, (420, 0))
        self.heat_map_win = Window("HeatMap", cv2.WINDOW_NORMAL, (840, 350))
        self.plot_win = Window("ResultsPlot", cv2.WINDOW_NORMAL, (0, 350))

        self.plot_win.resize(576, 331)
        
        self.frame_diff_canvas = Canvas(self._capture.width, self._capture.height, 1)
        self.vector_field_canvas = Canvas(self._capture.width, self._capture.height, 1, 255)
        self.plot_canvas = Canvas(700, 300, 3)

        self.grid = Grid(16, self._capture.width, self._capture.height)

        self.frame_diff = FrameDifference()
        self.heat_map = HeatMap()
        self.shift = Shift(200, 150, 100, 100)
        self.view = View()


    def frame_diff_control(self):
        """
        Controlling the frame differencing function.
        """
        self.frame_diff.apply_frame_difference(self._video, self.frame_diff_canvas)

    def grid_control(self):
        """
        Controlling vector field grid.
        """
        self.grid.calc_optical_flow(self._video)
        self.grid.calc_global_resultant_vector()
        self.grid.update_new_points_3D()
        self.grid.update_vector_lengths()

    def heat_map_control(self):
        """
        Controlling the motion heat-map function.
        """
        self.heat_map.calc_heat_map(self.grid, 10)
        self.heat_map.get_motion_points(self.grid, 10)
        self.heat_map.analyse_two_largest_points()

    def shift_control(self):
        """
        Controlling the shift function.
        """
        self.shift.calc_shift(self.grid, self._capture)

    def view_control(self):
        """
        Controlling the View.
        """
        self.view.show_heat_map(self.heat_map_win, self.heat_map)
        self.view.show_canvas(self.frame_diff_win, self.frame_diff_canvas)
        self.view.show_shift(self.shift, self._video)                     
        self.view.show_image(self.webcam_win, self._video.frame)
        self.view.show_vector_field(self.grid,
                                    self.vector_field_win,
                                    self.vector_field_canvas)
        self.view.show_global_vector_results(self.grid,
                                            self.plot_win,
                                            self.plot_canvas)

    def main_loop(self):
        """
        The main event loop
        """
        while True:
            self._video.get_frame(self._capture)
            if self._video.ret:

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

        self._capture.release()
        cv2.destroyAllWindows()

