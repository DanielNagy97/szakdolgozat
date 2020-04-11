import numpy as np
from arpt.widget import Widget


class Shift(Widget):
    """
    Shift widget representation
    """

    def __init__(self, position, dimension, image, speed, attenuation,
                 transparent):
        """
        Initialize new shift widget.
        :param position: position of the element tuple of (x,y)
        :param dimension: dimension of the element tuple of (width, height)
        :param image: source of the image file
        :param speed: the speed of the element. \
            Value bellow 1 means slower speed.
        :param attenuation: attenuation of the element. \
            The value should be smaller than 1 and not negative.
        """
        super().__init__(position, dimension, image, transparent)
        self.speed = speed
        self.attenuation = attenuation
        self.velocity = [0.0, 0.0]

    def calc_shift(self, grid, dimensions_of_frame):
        """
        Calculate the shift vectors.
        :param grid: grid object
        :param dimensions_of_frame: dimension of frame, tuple of (w, h)
        :return: None
        """
        x, y, w, h = np.uint8(np.floor(np.divide((*self._position,
                                                  *self._dimension),
                                                 grid.grid_step)))

        # NOTE: reducing computation time by using the grid's direction vectors
        # Calculating the local resultant vector for the Widget
        euclidean_vectors = np.reshape(grid.euclidean_vectors,
                                       grid.old_points_3D.shape)
        local_euclidean_vector = \
            np.mean(euclidean_vectors[y:y+h, x:x+w].reshape((-1, 2)),
                    axis=0)

        self.velocity = np.add(self.velocity,
                               np.multiply(local_euclidean_vector, self.speed))

        self._position = np.add(self._position, self.velocity)

        self.velocity = np.multiply(self.velocity, self.attenuation)

        cap_width, cap_height = dimensions_of_frame

        pos_x, pos_y = self._position
        width, height = self._dimension

        if pos_x + width >= cap_width:
            pos_x = cap_width - width

        if pos_y + height >= cap_height:
            pos_y = cap_height - height

        if pos_x < 0:
            pos_x = 0

        if pos_y < 0:
            pos_y = 0

        self._position = (pos_x, pos_y)
