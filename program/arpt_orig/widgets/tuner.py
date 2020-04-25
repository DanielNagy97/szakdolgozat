import cv2
from arpt.widget import Widget


class Tuner(Widget):
    """
    Tuner widget representation
    """
    def __init__(self, position, dimension, image, min_value=0, max_value=100):
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
        self.value = min_value
        self._original_image = self._image

        self.change = 360/(max_value-min_value)
        self.angle = 0

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

                    self.value += swirl.angles_of_rotation[i]/9
                    if self.value < self.min_value:
                        self.value = self.min_value
                    if self.value > self.max_value:
                        self.value = self.max_value

    def rotate_widget(self):
        angle = self.value*self.change

        center = (self._dimension[0]/2, self._dimension[1]/2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        self._image = cv2.warpAffine(self._original_image,
                                     M, self._dimension)
