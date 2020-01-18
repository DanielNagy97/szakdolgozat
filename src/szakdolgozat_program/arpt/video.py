import cv2
from .capture_device import CaptureDevice


class Video(object):
    """
    Class for providing a more convenient access for the video capture device
    """

    def __init__(self, source, dimension, to_flip=False):
        """
        Initialize the video
        :param source: index of the device, or file name in string
        :param dimension: The dimension of the frame, tuple of (width, height)
        :param to_flip: To mirror the image or not
        """
        self._capture = CaptureDevice(source, dimension)
        self._to_flip = to_flip
        self._dimension = dimension

        self.ret, self._frame = self._capture.read()
        if self._to_flip:
            self._frame = self.flip_frame(self._frame)

        if self._dimension != self._capture.dimension:
            self.resize_current_frame()

        self._gray_frame = self.make_gray_frame(self._frame)
        self._old_gray_frame = self._gray_frame

    def make_gray_frame(self, frame):
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

    def get_frame(self):
        """
        Get the next frame from the capture device.
        """
        self._old_gray_frame = self._gray_frame
        self.ret, self._frame = self._capture.read()
        if self._to_flip:
            self._frame = self.flip_frame(self._frame)

        if self._dimension != self._capture.dimension:
            self.resize_current_frame()

        self._gray_frame = self.make_gray_frame(self._frame)

    def resize_current_frame(self):
        """
        Resizing the current frame
        """
        self._frame = cv2.resize(self._frame, self._dimension, fx=0, fy=0,
                                 interpolation=cv2.INTER_CUBIC)

    def release_capture_device(self):
        """
        Releasing capture device
        """
        self._capture.release()

    @property
    def dimension(self):
        """
        Get the dimension of the frame.
        :return: tuple of (width, height)
        """
        return self._dimension

    @property
    def frame(self):
        """
        Get the current frame.
        """
        return self._frame

    @property
    def gray_frame(self):
        """
        Get the current grayscale frame.
        """
        return self._gray_frame

    @property
    def old_gray_frame(self):
        """
        Get the former grayscale frame.
        """
        return self._old_gray_frame
