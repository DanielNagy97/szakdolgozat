import numpy as np

class canvas():
    def __init__(self, height, width, channels,fill_value=0):
        self.canvas = np.zeros([height,width,channels],dtype=np.uint8)
        self.canvas.fill(fill_value)

    def make_canvas(self, height, width, channels):
        self.canvas = np.zeros([height,width,channels],dtype=np.uint8)

    def fill(self, value):
        self.canvas.fill(value)

    def update(self, new_canvas):
        self.canvas = new_canvas