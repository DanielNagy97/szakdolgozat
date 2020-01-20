import numpy as np
from arpt import vector as v
from arpt.widget import Widget


class Shift(Widget):
    """
    Shift vector field representation
    """

    def __init__(self, position, dimension, image):
        """
        Initialize new shift widget.
        :param position: position of the element tuple of (x,y)
        :param dimension: dimension of the element tuple of (width, height)
        :param image: source of the image file
        """
        super().__init__(position, dimension, image)
        self.velocity = [0.0, 0.0]

    def calc_shift(self, grid, dimensions_of_frame, speed, attenuation):
        """
        Calculate the shift vectors.
        :param grid: grid object
        :param dimensions_of_frame: dimension of frame, tuple of (w, h)
        :param speed: the speed of the element. \
            Value bellow 1 means slower speed.
        :param attenuation: attenuation of the element. \
            The value should be smaller than 1 and not negative.
        :return: None
        """
        x, y, w, h = np.uint8(np.floor(np.divide((*self._position,
                                                  *self._dimension),
                                                 grid.grid_step)))

        local_vector_sum = \
            np.array([grid.old_points_3D[y:y+h, x:x+w].sum(axis=0),
                      grid.new_points_3D[y:y+h, x:x+w].sum(axis=0)],
                     dtype=np.float32).sum(axis=1)

        local_direction_vector = v.get_direction_vector(local_vector_sum)

        vector_count = len(grid.old_points_3D[y:y+h, x:x+w].reshape(-1, 2))

        self.velocity = np.add(self.velocity,
                               np.multiply(np.divide(local_direction_vector,
                                                     vector_count),
                                           speed))

        # self.velocity = np.add(self.velocity,
        #                       np.divide(local_direction_vector,
        #                                 np.multiply(vector_count, speed)))

        self._position = np.add(self._position, self.velocity)

        self.velocity = np.multiply(self.velocity, attenuation)

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
