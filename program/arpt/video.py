import cv2


class Video(object):
    """
    Class for providing a more convenient access for the video capture device
    """
    
    def __init__(self, source, dimension=None, need_flip=False):
        """
        Initialize the video.
        :param source: index of the device as an integer, or a file path as a string
        :param dimension: the dimension of the frame, tuple of (width, height)
        :param need_flip: logical value which signs that the horizontal flipping is necessary or not
        """
        self._capture = cv2.VideoCapture(source)
        self._need_flip = need_flip
        self._dimension = dimension

    def get_next_frame(self):
        """
        Get the next frame from the capture.
        :return: NumPy array of a properly sized and flipped frame
        """
        has_read_properly, frame = self._capture.read()
        if has_read_properly is False:
            return None
        if self._dimension is not None:
            frame = cv2.resize(frame,
                               self._dimension,
                               fx=0,
                               fy=0,
                               interpolation=cv2.INTER_CUBIC)
        if self._need_flip:
            frame = cv2.flip(frame, 1)
        return frame

    def __enter__(self):
        """
        Returns with the Video instance as a context manager.
        """
        return self

    def __exit__(self, exc_type, value, traceback):
        """
        Release the capture device when exiting from the context.
        """
        self._capture.release()

