# Module for setting widget actions
import cv2


def placeholder(eze, aza):
    pass


def next_slide(controller, button_widget):
    if button_widget._pushed:
        controller.current_scene += 1
        button_widget._pushed = False


def prew_slide(controller, button_widget):
    if button_widget._pushed:
        controller.current_scene -= 1
        button_widget._pushed = False


def go_back_to_menu(controller, button_widget):
    if button_widget._pushed:
        controller.current_scene = 1
        button_widget._pushed = False


def alpha_slide(controller, button_widget):
    if button_widget._pushed:
        controller.current_scene = 3
        button_widget._pushed = False


def rgb_slide(controller, button_widget):
    if button_widget._pushed:
        controller.current_scene = 4
        button_widget._pushed = False


def threshold_slide(controller, button_widget):
    if button_widget._pushed:
        controller.current_scene = 5
        button_widget._pushed = False


to_treshold = False


def threshold_tuner(controller, tuner_widget):
    global to_treshold

    if to_treshold:
        value = tuner_widget.value
        ret, threshold = cv2.threshold(controller._video.gray_frame,
                                       value, 255, cv2.THRESH_BINARY)
        threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
        controller._video.frame = threshold


def toggle_threshold(controller, button_widget):
    global to_treshold
    if button_widget._pushed:
        to_treshold = not to_treshold
        button_widget._pushed = False


channel = 0


def toggle_channel(controller, button_widget):
    global channel
    if button_widget._pushed:
        channel += 1
        if channel == 4:
            channel = 0
        button_widget._pushed = False


def switch_color_channel(controller, button_widget):
    global channel
    if button_widget._pushed:

        if channel == 1:
            r = controller._video.frame.copy()
            # set blue and green channels to 0
            r[:, :, 0] = 0
            r[:, :, 1] = 0
            controller._video.frame = r

        elif channel == 2:
            g = controller._video.frame.copy()
            # set blue and red channels to 0
            g[:, :, 0] = 0
            g[:, :, 2] = 0
            controller._video.frame = g

        elif channel == 3:
            b = controller._video.frame.copy()
            # set green and red channels to 0
            b[:, :, 1] = 0
            b[:, :, 2] = 0
            controller._video.frame = b


to_alpha = False


def alpha_tuner(controller, tuner_widget):
    global to_alpha

    if to_alpha:
        value = tuner_widget.value
        controller._transparency = value
        cv2.putText(controller._video.frame,
                    "Alfa: " + str(format(value, '.2f')),
                    (100, 100),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, 255)


def toggle_alpha(controller, button_widget):
    global to_alpha
    if button_widget._pushed:
        to_alpha = not to_alpha
        button_widget._pushed = False