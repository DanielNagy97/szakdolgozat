import numpy as np


class Canvas(object):
    """
    Canvas representation
    """

    def __init__(self, width, height, channels, fill_value=0):
        """
        Initialize the canvas
        :param width: width of the canvas in pixels
        :param height: height of the canvas in pixels
        :param channels: number of the channels
        :param fill_value: initial values of the pixels
        """
        # TODO: Unify the order of parameters!
        # Parameters unified: width is always before height
        self.make_canvas(width, height, channels)
        self._canvas.fill(fill_value)

    def make_canvas(self, width, height, channels):
        """
        Create a new canvas.
        :param width: width of the canvas in pixels
        :param height: height of the canvas in pixels
        :param channels: number of the channels
        :return: None
        """
        # QUEST: Where is this necessary?
        self._canvas = np.zeros([height, width, channels], dtype=np.uint8)

    def fill(self, value):
        """
        Fill the values of the canvas.
        :param value: new value
        :return: None
        """
        self._canvas.fill(value)

    def update(self, new_canvas):
        """
        Update the managed canvas.
        :param new_canvas: the new managed canvas object
        :return: None
        """
        self._canvas = new_canvas

