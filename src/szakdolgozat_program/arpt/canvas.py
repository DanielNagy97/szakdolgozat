import numpy as np


class Canvas(object):
    """
    Canvas representation
    """

    def __init__(self, dimension, channels, fill_value=0):
        """
        Initialize the canvas
        :param dimension: tuple of (width, height) in pixels
        :param channels: number of the channels
        :param fill_value: initial values of the pixels
        """
        # TODO: Unify the order of parameters!
        # Parameters unified: width is always before height

        self._canvas = np.zeros([*dimension[::-1], channels], dtype=np.uint8)
        self._canvas.fill(fill_value)

    def fill(self, value):
        """
        Fill the values of the canvas.
        :param value: new value
        :return: None
        """
        self._canvas.fill(value)

    @property
    def canvas(self):
        """
        Get the canvas.
        :return: np array [height, width, channels]
        """
        return self._canvas

    @canvas.setter
    def canvas(self, new_canvas):
        """
        Update the managed canvas.
        :param new_canvas: the new managed canvas
        np array [height, width, channels]
        :return: None
        """
        self._canvas = new_canvas
