import cv2


class FrameDifference(object):
    """
    Difference of two sequent frame
    """

    @staticmethod
    def apply_frame_difference(video, canvas):
        """
        Update the image of the canvas by the thresholded absolute difference.
        :param video: video capture device
        :param canvas: the canvas object
        :return: None
        """
        # NOTE: The self is not used.
        # A function or a static method is better in this case.
        difference = cv2.absdiff(video.gray_frame, video.old_gray_frame)
        blurred = cv2.blur(difference, (20, 20))
        _, thresholded = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
        canvas.canvas = (cv2.addWeighted(canvas.canvas,
                                         0.9, thresholded, 1-0, 0))
