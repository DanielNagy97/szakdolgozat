import cv2

class cap():
    def __init__(self,index,width,height):
        self.cam = cv2.VideoCapture(index)
        self.conf_device(width,height)
        self.width, self.height = self.get_frame_dimensions()

    def conf_device(self,width,height):
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,width)

    def get_frame_dimensions(self):
        return int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        return self.cam.read()

    def release(self):
        self.cam.release()