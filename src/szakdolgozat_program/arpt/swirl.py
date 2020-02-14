import numpy as np


class Swirl(object):
    """
    Swirl class
    """
    def __init__(self):
        self._points = np.empty((0, 2), dtype=np.int32)

    def calc_swirl(self, grid, motion_blobs, video, eps=2):
        """
        Calculate the intersections of the vector field.
        :param grid: the grid object
        :param eps: the minimum lengths of the vectors
        :return: None
        """
        # NOTE: needs some kind of smoothing method

        # storing previous points to smoothen the result
        prev_points = np.empty((0, 2), dtype=np.int32)
        if self._points.any():
            prev_points = self._points
        self._points = np.empty((0, 2), dtype=np.int32)
        for blob in motion_blobs:
            (x, y, w, h) = blob

            direction_vectors = np.reshape(grid.direction_vectors,
                                           grid.old_points_3D.shape)
            direction_vectors = \
                direction_vectors[y:y+h, x:x+w].reshape((-1, 2))

            lenghts_new_shape = list(grid.old_points_3D.shape)
            lenghts_new_shape[-1] = 1
            vector_lenghts = np.reshape(grid.vector_lengths,
                                        tuple(lenghts_new_shape))
            vector_lenghts = vector_lenghts[y:y+h, x:x+w].reshape((-1, ))

            indexes = np.where(vector_lenghts > eps)

            direction_vectors = np.take(direction_vectors,
                                        indexes,
                                        axis=0).reshape(-1, 2)

            selected_old_points = \
                np.take(grid.old_points_3D[y:y+h, x:x+w].reshape(-1, 2),
                        indexes,
                        axis=0).reshape(-1, 2)

            b = np.multiply(direction_vectors,
                            selected_old_points)
            b = b.sum(axis=1)

            intersections = np.empty((0, 2), dtype=np.int32)
            for i in range(len(b)-1):
                x = np.linalg.solve([direction_vectors[i],
                                     direction_vectors[i+1]],
                                    [b[i], b[i+1]])
                intersections = np.append(intersections,
                                          np.array([x], dtype=np.int32),
                                          axis=0)

            if len(intersections) > 0:
                mean_point = np.array([np.mean(intersections, axis=0)],
                                      dtype=np.uint16)
                if ((mean_point[0][0] > video.dimension[0] or
                        mean_point[0][1] > video.dimension[1]) and
                   prev_points.any()):
                    self._points = prev_points
                else:
                    self._points = np.append(self._points, mean_point, axis=0)

    @property
    def points(self):
        return self._points
