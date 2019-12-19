import cv2

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
