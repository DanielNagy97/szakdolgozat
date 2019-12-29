import cv2

class video():
    def __init__(self, cap):
        self.ret, self.frame = cap.read()
        self.frame = self.flip_frame(self.frame)
        self.gray_frame = self.get_gray_frame(self.frame)
        self.old_gray_frame = self.gray_frame

    def get_gray_frame(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def flip_frame(self, frame):
        return cv2.flip(frame, 1)

    def get_frame(self, cap):
        self.old_gray_frame = self.gray_frame
        self.ret, self.frame = cap.read()
        self.frame = self.flip_frame(self.frame)
        self.gray_frame = self.get_gray_frame(self.frame)