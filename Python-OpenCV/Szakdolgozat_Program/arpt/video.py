import cv2

class video():
    def __init__(self,cap):
        self.ret, self._frame = cap.read()
        self._frame = self.flip_frame(self._frame)
        self._gray_frame = self.gray_frame(self._frame)
        self._old_gray_frame = self._gray_frame

    def gray_frame(self,frame):
        return cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    def flip_frame(self,frame):
        return cv2.flip(frame, 1)

    def get_frame(self, cap):
        self._old_gray_frame = self._gray_frame
        self.ret, self._frame = cap.read()
        self._frame = self.flip_frame(self._frame)
        self._gray_frame = self.gray_frame(self._frame)