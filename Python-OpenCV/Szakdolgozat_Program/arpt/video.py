import cv2


class Video(object):
    """
    Class for providing a more convenient access for the video capture device
    """

    def __init__(self, cap):
        """
        Initialize the video
        """
        self.ret, self.frame = cap.read()
        self.frame = self.flip_frame(self.frame)
        self.gray_frame = self.get_gray_frame(self.frame)
        self.old_gray_frame = self.gray_frame

    def get_gray_frame(self, frame):
        """
        Get the grayscale frame.
        :return: a grayscale image
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def flip_frame(self, frame):
        """
        Flip the frame.
        :return: an image object
        """
        return cv2.flip(frame, 1)

    def get_frame(self, cap):
        """
        Get the next frame from the capture device.
        """
        self.old_gray_frame = self.gray_frame
        self.ret, self.frame = cap.read()
        self.frame = self.flip_frame(self.frame)
        self.gray_frame = self.get_gray_frame(self.frame)

