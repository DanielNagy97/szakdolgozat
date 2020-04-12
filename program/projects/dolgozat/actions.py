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


def frame_diff_show(controller, button_widget):
    if button_widget._pushed:

        canvas = controller.frame_diff.canvas.canvas
        canvas = cv2.resize(canvas, button_widget._dimension, fx=0, fy=0,
                            interpolation=cv2.INTER_CUBIC)
        results_canvas = cv2.cvtColor(canvas,
                                      cv2.COLOR_GRAY2BGR)
        button_widget._image = results_canvas


def heatmap_show(controller, button_widget):
    if button_widget._pushed:

        canvas = controller.heat_map.map
        canvas = cv2.resize(canvas, button_widget._dimension, fx=0, fy=0,
                            interpolation=cv2.INTER_AREA)

        button_widget._image = canvas


def vector_field_show(controller, button_widget):
    if button_widget._pushed:
        controller.view.show_vector_field(controller.grid, controller.swirl,
                                          controller._canvasses['vectorfield'])
        canvas = controller._canvasses['vectorfield'].canvas
        canvas = cv2.resize(canvas, button_widget._dimension, fx=0, fy=0,
                            interpolation=cv2.INTER_CUBIC)
        results_canvas = cv2.cvtColor(canvas,
                                      cv2.COLOR_GRAY2BGR)
        button_widget._image = results_canvas


def symbol_show(controller, button_widget):
    if button_widget._pushed:

        canvas = controller._ocr.predicted_gest
        canvas = cv2.resize(canvas, button_widget._dimension, fx=0, fy=0,
                            interpolation=cv2.INTER_AREA)
        results_canvas = cv2.cvtColor(canvas,
                                      cv2.COLOR_GRAY2BGR)
        button_widget._image = results_canvas


def sweep_show(controller, button_widget):
    if button_widget._pushed:
        controller.view.show_result_plot(controller.grid,
                                         controller._canvasses['resultplot'])

        canvas = controller._canvasses['resultplot'].canvas
        canvas = cv2.resize(canvas, button_widget._dimension, fx=0, fy=0,
                            interpolation=cv2.INTER_CUBIC)

        button_widget._image = canvas


def grab_show(controller, button_widget):
    if button_widget._pushed:

        canvas = controller._grab.grab_image
        canvas = cv2.resize(canvas, button_widget._dimension, fx=0, fy=0,
                            interpolation=cv2.INTER_AREA)
        results_canvas = cv2.cvtColor(canvas,
                                      cv2.COLOR_GRAY2BGR)
        button_widget._image = results_canvas
