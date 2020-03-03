from arpt.widget import Widget


class Tuner(Widget):
    """
    Tuner widget representation
    """
    def __init__(self, position, dimension, image, min_value, max_value):
        """
        Initialize new Tuner widget.
        :param position: position of the element tuple of (x,y)
        :param dimension: dimension of the element tuple of (width, height)
        :param image: source of the image file
        :param min_value: Minimum value of the widget
        :param max_value: Maximum value of the widget
        """
        super().__init__(position, dimension, image)
        self.min_value = min_value
        self.max_value = max_value

    def update_value(self, swirl):
        """
        Updating the value of the Tuner widget
        :param swirl: The Swirl object
        """
        if swirl.points.any():
            width, height = self._dimension
            pos_x, pos_y = self._position

            for i in range(len(swirl.points)):
                rot_x, rot_y = swirl.points[i]

                if ((rot_x > pos_x and
                        rot_y > pos_y) and
                        (rot_x < pos_x + width) and
                        rot_y < pos_y + height):
                    print(swirl.points[i])
                    print(swirl.angles_of_rotation[i])
