import cv2


def action01(controller, button_widget):
    if button_widget._pushed:
        controller.current_scene += 1


def action02(controller, button_widget):
    # For test purposes
    # Pushing the button, will take us to the next "slide"
    if button_widget._pushed:
        print("Screenshot!")
        cv2.imwrite("screenshot.png", controller._video.frame)
        button_widget._pushed = False
