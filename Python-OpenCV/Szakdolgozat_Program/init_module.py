import cv2
import vector_field_module as vf
import numpy as np

def init_windows():
    cv2.namedWindow("test",cv2.WINDOW_NORMAL)
    cv2.namedWindow("vectorField",cv2.WINDOW_NORMAL)
    cv2.namedWindow("frameDiff",cv2.WINDOW_NORMAL)
    cv2.namedWindow("HeatMap",cv2.WINDOW_NORMAL)
    cv2.namedWindow("ResultsPlot",cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ResultsPlot', 576,331)

    cv2.moveWindow("test", 0, 0)
    cv2.moveWindow("frameDiff", 420, 0)
    cv2.moveWindow("vectorField", 840, 0)
    cv2.moveWindow("HeatMap", 840, 350)
    cv2.moveWindow("ResultsPlot", 0, 350)

def init_capture_device():
    cap = cv2.VideoCapture(0)

    #configuring capture device
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)

    #getting frame dimensions
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    return (cap,cap_height,cap_width)


def init_vector_field(cap_width,cap_height):
    grid_step = int(cap_width/16)
    return vf.vector_field_grid(grid_step,cap_width,cap_height)

def init_first_frame(cap):
    _, old_frame = cap.read()
    old_frame = cv2.flip(old_frame, 1)
    return cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)

def init_canvases(cap_height,cap_width):
    vector_field_canvas = np.zeros([cap_height,cap_width,1],dtype=np.uint8)
    vector_field_canvas.fill(255)

    frame_diff_canvas = np.zeros([cap_height,cap_width,1],dtype=np.uint8)

    plot_canvas = np.zeros([300,700,3],dtype=np.uint8)

    #if the grid_step is 16 and the aspect ratio is 16:9 then the vector field consists of 8*15 points
    heat_map_canvas = np.zeros([8,15,3],dtype=np.uint8)
    #heat_map_canvas = np.zeros([17,32,3],dtype=np.uint8)

    return (vector_field_canvas,frame_diff_canvas,plot_canvas,heat_map_canvas)