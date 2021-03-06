import cv2


class Widget(object):
    """
    Base class of the widgets
    """

    def __init__(self, position, dimension, image, transparent=True):
        """
        Initialize new widget.
        :param position: position of the element tuple of (x,y)
        :param dimension: dimension of the element tuple of (width, height)
        :param image: source of the image file
        """
        self._position = position
        self._dimension = dimension
        self._transparent = transparent
        if image[-3:] == "png" or image[-3:] == "PNG":
            self._image = cv2.imread(image, -1)
        else:
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

    @property
    def transparent(self):
        return self._transparent
