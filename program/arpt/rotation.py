import numpy as np


class Rotation(object):
    """
    Rotation class
    """

    def __init__(self):
        """
        Initalize the Rotation function.
        """
        self._points = np.empty((0, 2), dtype=np.int32)
        self._angles_of_rotation = np.empty((0, 1), dtype=np.int32)

    def calc_rotation_points(self, grid, motion_blobs, video, eps=2):
        """
        Calculate the intersections of the vector field.
        :param grid: the grid object
        :param motion_blobs: the motion blobs of the heat-map
        :param video: the video object
        :param eps: the minimum lengths of the vectors
        :return: None
        """
        self._points = np.empty((0, 2), dtype=np.int32)
        self._angles_of_rotation = np.empty((0, 1), dtype=np.int32)
        for blob in motion_blobs:

            euclidean_vectors, selected_old_points = \
                self.get_euclidean_vectors_of_rotation(blob, grid, eps)

            b = self.get_b_sides_of_eq(euclidean_vectors, selected_old_points)

            intersections = self.get_intersections(b, euclidean_vectors)

            if len(intersections) > 0:
                mean_point = np.array([np.mean(intersections, axis=0)],
                                      dtype=np.uint16)
                if (mean_point[0][0] < video.dimension[0] or
                        mean_point[0][1] < video.dimension[1]):
                    phi = self.calc_angles_of_rotation(euclidean_vectors)

                    self._angles_of_rotation = \
                        np.append(self._angles_of_rotation, phi)
                    self._points = np.append(self._points, mean_point, axis=0)

    def calc_angles_of_rotation(self, euclidean_vectors):
        """
        Calculate the angle of rotation
        :param euclidean_vectors: the direction vectors of the swirl
        :return: angle of rotation as float32
        """
        x, y = np.hsplit(euclidean_vectors, 2)
        phi = np.mean(np.arctan2(x, y)) * 180 / np.pi

        return phi

    def get_euclidean_vectors_of_rotation(self, blob, grid, eps):
        """
        Calculate the direction vectors of the rotation
        :param blob: the blob of motion
        :param grid: the grid object
        :param eps: the minimum lengths of the vectors
        :return: direction vectors, selected old points as np.ndarray
        """
        (x, y, w, h) = blob

        euclidean_vectors = np.reshape(grid.euclidean_vectors,
                                       grid.old_points_3D.shape)
        euclidean_vectors = \
            euclidean_vectors[y:y+h, x:x+w].reshape((-1, 2))

        lenghts_new_shape = list(grid.old_points_3D.shape)
        lenghts_new_shape[-1] = 1
        vector_lenghts = np.reshape(grid.vector_lengths,
                                    tuple(lenghts_new_shape))
        vector_lenghts = vector_lenghts[y:y+h, x:x+w].reshape((-1, ))

        indexes = np.where(vector_lenghts > eps)

        euclidean_vectors = np.take(euclidean_vectors,
                                    indexes,
                                    axis=0).reshape(-1, 2)

        selected_old_points = \
            np.take(grid.old_points_3D[y:y+h, x:x+w].reshape(-1, 2),
                    indexes,
                    axis=0).reshape(-1, 2)

        return euclidean_vectors, selected_old_points

    def get_b_sides_of_eq(self, euclidean_vectors, selected_old_points):
        """
        Make the 'b' sides of equations
        :param euclidean_vectors: the direction vectors of the swirl
        :param selected_old_points: the selected original points
        :return: 'b' sides of equations as np.ndarray
        """
        b = np.multiply(euclidean_vectors,
                        selected_old_points)
        b = b.sum(axis=1)

        return b

    def get_intersections(self, b, euclidean_vectors):
        """
        Calculate the intersections of the swirl
        :param b: the b sides of the equations
        :param euclidean_vectors: the direction vectors of the swirl
        :return: intersections as np.ndarray
        """
        intersections = np.empty((0, 2), dtype=np.int32)
        for i in range(len(b)-1):
            x = np.linalg.solve([euclidean_vectors[i],
                                 euclidean_vectors[i+1]],
                                [b[i], b[i+1]])
            intersections = np.append(intersections,
                                      np.array([x], dtype=np.int32),
                                      axis=0)

        return intersections

    @property
    def points(self):
        return self._points

    @property
    def angles_of_rotation(self):
        return self._angles_of_rotation
