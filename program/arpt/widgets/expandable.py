import numpy as np
import arpt.vector as v
from arpt.widget import Widget


class Expandable(Widget):
    """
    Expand widget representation
    """
    def __init__(self, position, dimension, image, speed, attenuation,
                 header_size, transparent):
        """
        Initialize new Expandable widget.
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
        self.velocity = 0.0
        self.min_height = header_size
        self.original_image = self._image
        self.original_height = self._dimension[1]
        self._dimension = np.asarray(self._dimension)
        self._dimension[1] = self.min_height
        self._image = \
            self.original_image[self.original_height-self._dimension[1]:
                                self.original_height,
                                0:self.dimension[0]].copy()

    def calc_expandable(self, grid, dimensions_of_frame):
        """
        Calculate the Expandable's behaviour.
        :param grid: grid object
        :param dimensions_of_frame: dimension of frame, tuple of (w, h)
        :return: None
        """
        x, y, w, h = np.uint8(np.floor(np.divide((*self._position,
                                                  self._dimension[0],
                                                  self._dimension[1]),
                                                 grid.grid_step)))

        local_vector_sum = \
            np.array([grid.old_points_3D[y:y+h, x:x+w].sum(axis=0),
                      grid.new_points_3D[y:y+h, x:x+w].sum(axis=0)],
                     dtype=np.float32).sum(axis=1)

        local_euclidean_vector = v.get_euclidean_vector(local_vector_sum)

        vector_count = len(grid.old_points_3D[y:y+h, x:x+w].reshape(-1, 2))

        if local_euclidean_vector[1] != 0.0:
            self.velocity = \
                np.add(self.velocity,
                       np.multiply(np.divide(local_euclidean_vector[1],
                                             vector_count),
                                   self.speed))
        else:
            self.velocity = 0

        self._dimension[1] += int(self.velocity)

        self.velocity = np.multiply(self.velocity, self.attenuation)

        height = self.original_height

        if self._dimension[1] < self.min_height:
            self._dimension[1] = self.min_height

        if self._dimension[1] > height:
            self._dimension[1] = height

        if self._position[1]+self._dimension[1] > dimensions_of_frame[1]:
            self.dimension[1] = dimensions_of_frame[1]-self._position[1]

        self._image = \
            self.original_image[height-self._dimension[1]:height,
                                0:self.dimension[0]].copy()

    @property
    def actual_height(self):
        return self._dimension[1]
