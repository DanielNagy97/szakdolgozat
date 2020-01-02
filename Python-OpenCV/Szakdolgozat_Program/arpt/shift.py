import numpy as np

from arpt import vector as v

class Shift(object):
    """
    Shift vector field representation
    """

    def __init__(self, pos_x, pos_y, width, height):
        """
        Initialize new shift vector field.
        :param pos_x:
        :param pos_y:
        :param width:
        :param height:
        """
        self._pos_x = pos_x
        self._pos_y = pos_y
        self._width = width
        self._height = height
        self.velocity_x = 0.0
        self.velocity_y = 0.0

    def calc_shift(self, grid, dimensions_of_frame):
        """
        Calculate the shift vectors.
        :param grid: grid object
        :param dimensions_of_frame: dimension of frame as a tuple of (width, height)
        :return: None
        """
        x,y,w,h = np.uint8(np.floor(np.divide(( self._pos_x,
                                                self._pos_y,
                                                self._width,
                                                self._height), grid.grid_step)))

        local_vector_sum = np.array([grid.old_points_3D[y:y+h, x:x+w].sum(axis=0),
                                    grid.new_points_3D[y:y+h, x:x+w].sum(axis=0)],
                                    dtype=np.float32).sum(axis=1)

        local_direction_vector = v.get_direction_vector(local_vector_sum)

        vector_count = len(local_vector_sum)

        self.velocity_x += local_direction_vector[0] / vector_count*0.5
        self.velocity_y += local_direction_vector[1] / vector_count*0.5

        self._pos_x += self.velocity_x
        self._pos_y += self.velocity_y

        self.velocity_x *= 0.8
        self.velocity_y *= 0.8

        cap_width, cap_height = dimensions_of_frame

        if self._pos_x + self._width >= cap_width:
            self._pos_x = cap_width - self._width

        if self._pos_y + self._height >= cap_height:
            self._pos_y = cap_height - self._height

        if self._pos_x < 0:
            self._pos_x = 0

        if self._pos_y < 0:
            self._pos_y = 0

    @property
    def pos_x(self):
        return self._pos_x

    @pos_x.setter 
    def pos_x(self, new_x_position):
        self._pos_x = new_x_position

    @property
    def pos_y(self):
        return self._pos_y

    @pos_y.setter 
    def pos_y(self, new_y_position):
        self._pos_y = new_y_position

    @property
    def height(self):
        return self._height

    @height.setter 
    def height(self, new_height):
        self._height = new_height

    @property
    def width(self):
        return self._width

    @width.setter 
    def width(self, new_width):
        self._width = new_width