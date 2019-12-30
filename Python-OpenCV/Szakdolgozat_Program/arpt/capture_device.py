import cv2

class CaptureDevice(object):
    """
    Capture device representation
    """

    def __init__(self, index, width, height):
        """
        Initialize a new capture device.
        :param index: index of the device
        :param width: the desired width of the frame in pixels
        :param height: the desired height of the frame in pixels
        """
        # TODO: It should be clarified the purpose of width and height!
        # The width and height in params are desired values only, the final values depends on the device itself
        self._camera = cv2.VideoCapture(index)
        self.configure_device(width, height)
        self.width, self.height = self.get_frame_dimensions()

        if self.width != width or self.height != height:
            print("""[CaptureDevice]: The desired resolution (%d, %d) is NOT available,
                 the device will run with (%d, %d) resolution!""" %(width, height, self.width, self.height))

            
    def configure_device(self, width, height):
        """
        Configure the device.
        :param width: width of the frame in pixels
        :param height: height of the frame in pixels
        :return: None
        """
        # NOTE: The usage of dimensions setter property may be better!
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    def get_frame_dimensions(self):
        """
        Get the dimension of the frame.
        :return: tuple of (width, height)
        """
        # TODO: Use getter property!
        return  int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

