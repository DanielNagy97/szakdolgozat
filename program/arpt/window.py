import cv2


class Window(object):
    """
    Window representation
    """

    def __init__(self, name, mode, position=(0, 0)):
        """
        Initialize a new window.
        :param name: unique name of the window
        :param mode: window mode
        :param pos_x: X position
        :param pos_y: Y position
        """
        self.named_window = cv2.namedWindow(name, mode)
        self._name = name
        self.move(position)

    def resize(self, width, height):
        """
        Resize the window.
        :param width: new width of the window
        :param height: new height of the window
        :return: None
        """
        cv2.resizeWindow(self._name, width, height)

    def move(self, position):
        """
        Move the window to the given position.
        :param position: new position of the window as a tuple of (x, y)
        :return: None
        """
        cv2.moveWindow(self._name, *position)

    @property
    def name(self):
        return self._name
