import cv2

import vector_field_module as vf
import frame_diff_module as fd

from arpt.grid import grid
from arpt.cap_device import cap
from arpt.video import video
from arpt.view import view
from arpt.frame_diff import frame_diff
from arpt.canvas import canvas
from arpt.window import window
from arpt.heat_map import heat_map

cap = cap(0,640,360)

video = video(cap)

view = view()

webcam_win = window("test",cv2.WINDOW_NORMAL,0,0)
vector_field_win = window("vectorField",cv2.WINDOW_NORMAL,840,0)
frame_diff_win = window("frameDiff",cv2.WINDOW_NORMAL,420,0)
heat_map_win = window("HeatMap",cv2.WINDOW_NORMAL,840,350)
plot_win = window("ResultsPlot",cv2.WINDOW_NORMAL,0,350)
plot_win.resize(576,331)

frame_diff_canvas = canvas(cap.height,cap.width,1)
vector_field_canvas = canvas(cap.height,cap.width,1,255)
plot_canvas = canvas(300,700,3)
heat_map_canvas = canvas(8,15,3)

grid = grid(16,cap.width,cap.height)

frame_diff = frame_diff()

heat_map = heat_map()

if __name__ == "__main__":
    while True:
        video.get_frame(cap)

        if video.ret:
            
            frame_diff.frame_differencing(video, frame_diff_canvas)

            view.show_canvas(frame_diff_win, frame_diff_canvas)

            grid.calc_optical_flow(video)

            view.show_vector_field(grid, vector_field_win, vector_field_canvas)

            vf.global_resultant_vector(grid.old_points, grid.new_points,plot_canvas.canvas)
            vf.heat_map(grid.old_points,grid.new_points, heat_map_canvas.canvas)

            heat_map.calc_heat_map(grid)
            heat_map.get_motion_points(grid)

            #frame_diff_canvas.canvas = fd.frame_differencing(video._gray_frame, video._old_gray_frame, frame_diff_canvas.canvas)

            cv2.imshow("asd",heat_map.map)
            #view.show_canvas(webcam_win, video._frame)
            #view.show_canvas(vector_field_win.name, vector_field_canvas.canvas)
            #view.show_canvas(frame_diff_win.name, frame_diff_canvas.canvas)
            view.show_canvas(plot_win, plot_canvas)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()