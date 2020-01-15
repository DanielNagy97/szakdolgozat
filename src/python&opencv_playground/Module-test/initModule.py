import cv2


def init():
    init_windows()


def init_windows():
    cv2.namedWindow("test")
    cv2.namedWindow("vectorField")
    cv2.namedWindow("frameDiff")

