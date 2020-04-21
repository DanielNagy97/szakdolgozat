# Module for setting widget actions
import cv2


def jump_to(controller, button_widget):
    if button_widget.pushed:
        controller.current_scene = int(button_widget.arg)
        button_widget.pushed = False


threshold_value = 0.0


def threshold_tuner(controller, tuner_widget):
    global threshold_value

    threshold_value = tuner_widget.value


def toggle_threshold(controller, button_widget):
    global threshold_value
    if button_widget._pushed:
        ret, threshold = cv2.threshold(controller._video.gray_frame,
                                       threshold_value, 255, cv2.THRESH_BINARY)
        threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
        controller._video.frame = threshold
        cv2.putText(controller._video.frame,
                    "Threshold value: " + str(format(threshold_value, '.2f')),
                    (300, 50),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, 255)


channel = 0


def switch_color_channel(controller, button_widget):
    global channel
    if button_widget._pushed:
        channel += 1
        if channel == 4:
            channel = 0
        button_widget._pushed = False

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


alpha_value = 0.0


def alpha_tuner(controller, tuner_widget):
    global alpha_value
    alpha_value = tuner_widget.value


def toggle_alpha(controller, button_widget):
    global alpha_value
    if button_widget._pushed:
        controller._transparency = alpha_value
        cv2.putText(controller._video.frame,
                    "Opacity: " + str(format(alpha_value, '.2f')),
                    (300, 50),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, 255)


edge_value = 0


def toggle_edge(controller, button_widget):
    global edge_value
    if button_widget._pushed:
        output = cv2.Canny(controller._video.gray_frame, edge_value, 120)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        controller._video.frame = output


def edge_tuner(controller, tuner_widget):
    global edge_value
    edge_value = tuner_widget.value


def gauss_button(controller, button_widget):
    if button_widget._pushed:
        controller._video.frame = cv2.GaussianBlur(controller._video.frame,
                                                   (19, 19),
                                                   cv2.BORDER_DEFAULT)


def histeqv_button(controller, button_widget):
    gray = cv2.cvtColor(controller._video.frame, cv2.COLOR_BGR2GRAY)

    if button_widget._pushed:
        gray_frame = cv2.equalizeHist(gray)
        output = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    else:
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    controller._video.frame = output
