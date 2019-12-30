import cv2


class CaptureDevice(object):
    """
    Capture device representation
    """

    def __init__(self, index, desired_dimension):
        """
        Initialize a new capture device.
        :param index: index of the device
        :param desired_dimension: the desired dimension of the frame in pixels, a tuple of (width, height)
        """
        self._camera = cv2.VideoCapture(index)
        self.dimension = desired_dimension

        #if self.width != desired_dimension[0] or self.height != desired_dimension[1]:
        #    print("""[CaptureDevice]: The desired resolution (%d, %d) is NOT available,
        #         the device will run with (%d, %d) resolution!""" %(*desired_dimension, self.width, self.height))

    @property
    def dimension(self):
        """
        Get the dimension of the frame.
        :return: tuple of (width, height)
        """
        return self._dimension
        
    @dimension.setter 
    def dimension(self, desired_dimension):
        """
        Configure the device.
        Set the dimension depending on the capture device's available resolutions
        :param desired_dimension: tuple of (width, height) in pixels
        :return: None
        """
        width, height = desired_dimension
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._dimension = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        """
        Read a frame from the capture device.
        :return: a frame
        """
        return self._camera.read()

    def release(self):
        """
        Release the capture device.
        :return: None
        """
        # NOTE: Probably it is better to use a with context manager!
        self._camera.release()