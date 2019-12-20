import cv2
import numpy as np

class view():
    def __init__(self):
        pass

    def show_canvas(self, win, canvas):
        cv2.imshow(win.name,canvas.canvas)

    def show_vector_field(self, grid, win, canvas):
        canvas.fill(255)
        for k in range(int(grid.new_points.size/2)):
            current_vector = np.subtract(grid.new_points[k],grid.old_points[k])
            if abs(current_vector[0]) >= 2 or abs(current_vector[1]) >= 2:
                cv2.arrowedLine(canvas.canvas, tuple(grid.old_points[k]), tuple(grid.new_points[k]), (0,0,0), 2)
        self.show_canvas(win,canvas)