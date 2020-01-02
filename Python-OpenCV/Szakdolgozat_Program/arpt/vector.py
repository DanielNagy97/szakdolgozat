import numpy as np

def get_vector_lenght(vector):
    """
    Get the lenght of the vector.
    :param vector: np nd.array
    :return: Number, the lenght of the vector
    """
    return np.sqrt(np.sum(np.power(vector, 2)))

def get_normalized_vector(vector):
    """
    Get the normalized form of the vector.
    :param vector: np array with two elements
    :return: np array with two elements, normalized vector
    """
    return np.divide(vector, get_vector_lenght(vector))

def get_direction_vector(vector):
    """
    Get direction vector from the vector.
    :param vector: np nd.array
    :return: np array with two elements, direction vector
    """
    return np.subtract(vector[1], vector[0])