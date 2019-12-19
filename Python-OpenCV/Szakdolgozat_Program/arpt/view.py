import cv2

class view():
    def __init__(self):
        pass

    def show_canvas(self,win_name,canvas):
        cv2.imshow(win_name,canvas)

    def show_vector_field(self, old_points, new_points, canvas):
        pass