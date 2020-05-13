# Module for setting widget actions
import cv2


def placeholder(eze, aza):
    pass


def jump_to(controller, button_widget):
    if button_widget.pushed:
        controller.current_scene = int(button_widget.arg)
        button_widget.pushed = False


def frame_diff_show(controller, button_widget):
    if button_widget.pushed or button_widget.arg == "default":

        canvas = controller.frame_diff.canvas.canvas.copy()
        canvas = cv2.resize(canvas, button_widget.dimension, fx=0, fy=0,
                            interpolation=cv2.INTER_CUBIC)
        results_canvas = cv2.cvtColor(canvas,
                                      cv2.COLOR_GRAY2BGR)
        button_widget._image = results_canvas


def heatmap_show(controller, button_widget):
    if button_widget.pushed or button_widget.arg == "default":

        canvas = controller.heat_map.map.copy()
        canvas = cv2.resize(canvas, button_widget.dimension, fx=0, fy=0,
                            interpolation=cv2.INTER_AREA)

        button_widget._image = canvas


def vector_field_show(controller, button_widget):
    if button_widget.pushed or button_widget.arg == "default":
        controller.view.show_vector_field(controller.grid,
                                          controller._rotation,
                                          controller._canvasses['vectorfield'])
        canvas = controller._canvasses['vectorfield'].canvas
        canvas = cv2.resize(canvas, button_widget.dimension, fx=0, fy=0,
                            interpolation=cv2.INTER_CUBIC)
        results_canvas = cv2.cvtColor(canvas,
                                      cv2.COLOR_GRAY2BGR)
        button_widget._image = results_canvas


def symbol_show(controller, button_widget):
    if button_widget.pushed or button_widget.arg == "default":

        canvas = controller._symbol.predicted_gest
        canvas = cv2.resize(canvas, button_widget.dimension, fx=0, fy=0,
                            interpolation=cv2.INTER_AREA)
        results_canvas = cv2.cvtColor(canvas,
                                      cv2.COLOR_GRAY2BGR)
        button_widget._image = results_canvas


def symbol_canvas_show(controller, button_widget):
    if button_widget.pushed or button_widget.arg == "default":
        canvas = controller._symbol.canvas.canvas.copy()
        canvas = cv2.resize(canvas, button_widget.dimension, fx=0, fy=0,
                            interpolation=cv2.INTER_AREA)
        results_canvas = cv2.cvtColor(canvas,
                                      cv2.COLOR_GRAY2BGR)
        button_widget._image = results_canvas


def sweep_show(controller, button_widget):
    if button_widget.pushed or button_widget.arg == "default":
        controller.view.show_result_plot(controller.grid,
                                         controller._canvasses['resultplot'])

        canvas = controller._canvasses['resultplot'].canvas
        canvas = cv2.resize(canvas, button_widget.dimension, fx=0, fy=0,
                            interpolation=cv2.INTER_CUBIC)

        button_widget._image = canvas


def grab_show(controller, button_widget):
    if button_widget.pushed or button_widget.arg == "default":

        canvas = controller._blink.blink_image.copy()
        canvas = cv2.resize(canvas, button_widget.dimension, fx=0, fy=0,
                            interpolation=cv2.INTER_AREA)
        results_canvas = cv2.cvtColor(canvas,
                                      cv2.COLOR_GRAY2BGR)
        button_widget._image = results_canvas
