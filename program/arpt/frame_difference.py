import cv2
from arpt.canvas import Canvas


class FrameDifference(object):
    """
    Difference of two sequent frame
    """
    def __init__(self, video):
        self._canvas = Canvas(video.dimension, 1)

    def apply_frame_difference(self, video):
        """
        Update the image of the canvas by the thresholded absolute difference.
        :param video: video capture device
        :return: None
        """
        difference = cv2.absdiff(video.gray_frame, video.old_gray_frame)
        blurred = cv2.blur(difference, (20, 20))
        _, thresholded = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
        self._canvas.canvas = (cv2.addWeighted(self._canvas.canvas,
                                               0.9, thresholded, 1-0, 0))

    @property
    def canvas(self):
        return self._canvas
