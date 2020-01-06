import numpy as np
import arpt.vector as v

class Swirl(object):
    """
    Swirl class
    """
    def __init__(self):
        self.points = np.empty((0,2),dtype=np.int32)

    def calc_swirl(self, grid, eps=2):
        """
        Calculate the intersections of the vector field.
        :param grid: the grid object
        :param eps: the minimum lenghts of the vectors
        :return: None
        """
        direction_vectors=np.empty((0,2),dtype=np.float32)
        root_points=np.empty((0,1))
        for k in range(len(grid.new_points)):
            current_vector = np.subtract(grid.new_points[k], grid.old_points[k])
            if v.get_vector_lenght(current_vector) >= eps:
                direction_vectors = np.append(direction_vectors, np.array([current_vector]),axis=0)
                se = np.sum(np.multiply(current_vector,grid.old_points[k]))
                root_points = np.append(root_points, se)

        #NOTE: In this form it is very slow. It gets slower with more vectors... Have to find another solution.
        self.points = np.empty((0,2),dtype=np.int32)
        for i in range(len(root_points)):
            for j in range(i+1,len(direction_vectors)-1):
                x = np.linalg.solve([direction_vectors[i],direction_vectors[j]],[root_points[i],root_points[j]])
                self.points = np.append(self.points,np.array([x],dtype=np.int32),axis=0)