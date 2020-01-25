import numpy as np
import arpt.vector as v
from arpt.widget import Widget


class Expand(Widget):
    """
    Expand widget representation
    """
    def __init__(self, position, dimension, image):
        """
        Initialize new expandable widget.
        :param position: position of the element tuple of (x,y)
        :param dimension: dimension of the element tuple of (width, height)
        :param image: source of the image file
        """
        super().__init__(position, dimension, image)
        self.velocity = 0.0
        self._actual_height = self.dimension[1]

    def calc_expand(self, grid, dimensions_of_frame, speed, attenuation):
        x, y, w, h = np.uint8(np.floor(np.divide((*self._position,
                                                  self._dimension[0],
                                                  self._actual_height),
                                                 grid.grid_step)))

        local_vector_sum = \
            np.array([grid.old_points_3D[y:y+h, x:x+w].sum(axis=0),
                      grid.new_points_3D[y:y+h, x:x+w].sum(axis=0)],
                     dtype=np.float32).sum(axis=1)

        local_direction_vector = v.get_direction_vector(local_vector_sum)

        vector_count = len(grid.old_points_3D[y:y+h, x:x+w].reshape(-1, 2))

        self.velocity = np.add(self.velocity,
                               np.multiply(np.divide(local_direction_vector[1],
                                                     vector_count),
                                           speed))

        self._actual_height += self.velocity

        self.velocity = np.multiply(self.velocity, attenuation)

        width, height = self._dimension

        if self._actual_height < 50:
            self._actual_height = 50

        if self._actual_height > height:
            self._actual_height = height

    @property
    def actual_height(self):
        return self._actual_height
