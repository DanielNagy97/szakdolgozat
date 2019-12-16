import cv2
import vector_field_module as vf
import frame_diff_module as fd
import init_module as init

init.init_windows()
cap,cap_height,cap_width = init.init_capture_device()
old_points = init.init_vector_field(cap_width,cap_height)
old_gray_frame = init.init_first_frame(cap)
vector_field_canvas,frame_diff_canvas,plot_canvas,heat_map_canvas = init.init_canvases(cap_height,cap_width)

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        new_points, status, error = vf.calc_optical_flow(old_gray_frame,gray_frame,old_points)

        vf.draw_vector_field(old_points,new_points,vector_field_canvas)
        vf.global_resultant_vector(old_points,new_points,plot_canvas)
        vf.heat_map(old_points,new_points,heat_map_canvas)

        frame_diff_canvas = fd.frame_differencing(gray_frame,old_gray_frame,frame_diff_canvas)

        old_gray_frame = gray_frame.copy()

        cv2.imshow('test', frame)
        cv2.imshow('vectorField',vector_field_canvas)
        cv2.imshow('frameDiff',frame_diff_canvas)
        cv2.imshow('ResultsPlot',plot_canvas)
        #cv2.imshow("HeatMap",heat_map_canvas)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()