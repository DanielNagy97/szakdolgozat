import cv2

class window():
    def __init__(self,win_name, mode,pos_x=0,pos_y=0):
        self.named_window = cv2.namedWindow(win_name, mode)
        self.name = win_name
        self.mode = mode
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.move(pos_x,pos_y)
    
    def resize(self, width, height):
        cv2.resizeWindow(self.name, width, height)

    def move(self, pos_x, pos_y):
        cv2.moveWindow(self.name, pos_x, pos_y)