import numpy as np
import arpt.vector as v

class Swirl(object):
    """
    Swirl class
    """
    def __init__(self):
        self._points = np.empty((0,2),dtype=np.int32)

    def calc_swirl(self, grid, eps=3):
        """
        Calculate the intersections of the vector field.
        :param grid: the grid object
        :param eps: the minimum lenghts of the vectors
        :return: None
        """
        #NOTE:It's only working for one swirl point only and needs linear interpolation or some other smoothing methods!
        direction_vectors=np.empty((0,2),dtype=np.float32)
        b=np.empty((0,1))
        for k in range(len(grid.new_points)):
            current_vector = np.subtract(grid.new_points[k], grid.old_points[k])
            if v.get_vector_lenght(current_vector) >= eps:
                direction_vectors = np.append(direction_vectors, np.array([current_vector]),axis=0)
                b = np.append(b, np.sum(np.multiply(current_vector,grid.old_points[k])))

        intersections = np.empty((0,2),dtype=np.int32)

        for i in range(len(b)-1):
            x = np.linalg.solve([direction_vectors[i],direction_vectors[i+1]],[b[i],b[i+1]])
            intersections = np.append(intersections,np.array([x],dtype=np.int32),axis=0)

        if len(intersections) > 0:
            self._points = np.uint16(np.divide(intersections.sum(axis=0),len(intersections)))

    @property
    def points(self):
        return self._points    