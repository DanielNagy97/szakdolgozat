import cv2
import vector_field_module as vf
import frame_diff_module as fd
import init_module as init

from arpt.grid import grid
from arpt.cap_device import cap
from arpt.video import video
from arpt.view import view
from arpt.frame_diff import frame_diff
from arpt.canvas import canvas

cap = cap(0,640,360)

video = video(cap)

init.init_windows()

frame_diff_canvas = canvas(cap.height,cap.width,1)
vector_field_canvas = canvas(cap.height,cap.width,1,255)
plot_canvas = canvas(300,700,3)
heat_map_canvas = canvas(8,15,3)

grid = grid(16,cap.width,cap.height)

if __name__ == "__main__":
    while True:
        video.get_frame(cap)

        if video.ret:
            
            grid.calc_optical_flow(video._old_gray_frame, video._gray_frame)

            vf.draw_vector_field(grid.old_points, grid.new_points,vector_field_canvas.canvas)

            vf.global_resultant_vector(grid.old_points, grid.new_points,plot_canvas.canvas)
            vf.heat_map(grid.old_points,grid.new_points, heat_map_canvas.canvas)

            frame_diff_canvas.canvas = fd.frame_differencing(video._gray_frame, video._old_gray_frame, frame_diff_canvas.canvas)

            cv2.imshow('test', video._frame)
            cv2.imshow('vectorField',vector_field_canvas.canvas)
            cv2.imshow('frameDiff',frame_diff_canvas.canvas)
            cv2.imshow('ResultsPlot',plot_canvas.canvas)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()