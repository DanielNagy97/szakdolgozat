import cv2
from arpt.grid import grid
from arpt.cap_device import cap
from arpt.video import video
from arpt.view import view
from arpt.frame_diff import frame_diff
from arpt.canvas import canvas
from arpt.window import window
from arpt.heat_map import heat_map
from arpt.shift import shift

class controller():
    def __init__(self):
        self.cap = cap(0, 640, 360)

        self.video = video(self.cap)

        self.webcam_win = window("test", cv2.WINDOW_NORMAL, 0, 0)
        self.vector_field_win = window("vectorField", cv2.WINDOW_NORMAL, 840, 0)
        self.frame_diff_win = window("frameDiff", cv2.WINDOW_NORMAL, 420, 0)
        self.heat_map_win = window("HeatMap", cv2.WINDOW_NORMAL, 840, 350)
        self.plot_win = window("ResultsPlot", cv2.WINDOW_NORMAL, 0, 350)
        self.resize_window(self.plot_win, 576, 331)

        self.frame_diff_canvas = canvas(self.cap.height, self.cap.width, 1)
        self.vector_field_canvas = canvas(self.cap.height, self.cap.width, 1, 255)
        self.plot_canvas = canvas(300, 700, 3)

        self.grid = grid(16, self.cap.width, self.cap.height)

        self.frame_diff = frame_diff()

        self.heat_map = heat_map()

        self.shift = shift(200, 150, 100, 100)

        self.view = view()

    def resize_window(self, win, width, heigth):
        win.resize(width,heigth)

    def frame_diff_control(self):
        self.frame_diff.frame_differencing(self.video, self.frame_diff_canvas)

    def grid_control(self):
        self.grid.calc_optical_flow(self.video)
        self.grid.calc_global_resultant_vector()

    def heat_map_control(self):
        self.heat_map.calc_heat_map(self.grid)
        self.heat_map.get_motion_points(self.grid)
        self.heat_map.analyse_two_largest_points()

    def shift_control(self):
        self.shift.calc_shift(self.grid, self.cap)

    def view_control(self):
        self.view.show_heat_map(self.heat_map_win, self.heat_map)
        self.view.show_canvas(self.frame_diff_win, self.frame_diff_canvas)
        self.view.show_shift(self.shift, self.video)                     
        self.view.show_image(self.webcam_win, self.video.frame)
        self.view.show_vector_field(self.grid,
                                    self.vector_field_win,
                                    self.vector_field_canvas)
        self.view.show_global_vector_results(self.grid,
                                            self.plot_win,
                                            self.plot_canvas)

    def main_loop(self):
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