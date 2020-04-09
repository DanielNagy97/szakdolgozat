import cv2
import numpy as np


def action01(controller, button_widget):
    if button_widget._pushed:
        controller.current_scene += 1


def action02(controller, button_widget):
    if button_widget._pushed:
        print("Screenshot!")
        cv2.imwrite("screenshot.png", controller._video.frame)
        button_widget._pushed = False


def threshold_tuner(controller, tuner_widget):
    value = np.uint8(np.round(tuner_widget.value))

    ret, threshold = cv2.threshold(controller._video.gray_frame,
                                   value, 255, cv2.THRESH_BINARY)

    threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
    controller._video.frame = threshold
