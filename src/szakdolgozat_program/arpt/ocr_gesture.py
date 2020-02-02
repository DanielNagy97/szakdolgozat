import cv2


class Ocr_gesture(object):
    def __init__(self, heat_map):
        pass

    def draw_gesture(self, heat_map, canvas):
        canvas.fill(255)
        i = 1
        while i < len(heat_map.motion_points_roots):
            cv2.line(canvas.canvas,
                     tuple(heat_map.motion_points_roots[i-1]),
                     tuple(heat_map.motion_points_roots[i]),
                     0,
                     1)
            i += 1
