import cv2


class Widget(object):
    def __init__(self, position, dimension, image):
        """
        Initialize new widget.
        :param position: position of the element tuple of (x,y)
        :param dimension: dimension of the element tuple of (width, height)
        :param image: source of the image file
        """
        self._position = position
        self._dimension = dimension
        self._image = cv2.imread(image)
        self._image = cv2.resize(self._image, self._dimension,
                                 interpolation=cv2.INTER_CUBIC)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_position):
        self._position = new_position

    @property
    def dimension(self):
        return self._dimension

    @dimension.setter
    def dimension(self, new_dimension):
        self._dimension = new_dimension

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, new_image):
        self._image = new_image
