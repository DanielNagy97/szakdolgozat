import cv2

class cap():
    def __init__(self,index,width,height):
        self.cam = cv2.VideoCapture(index)
        self.conf_device(width,height)

        #getting frame dimensions
        self.height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))

    def conf_device(self,width,height):
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,width)

    def read(self):
        return self.cam.read()

    def release(self):
        self.cam.release()