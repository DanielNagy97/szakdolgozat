import numpy as np


def get_vector_length(vector):
    """
    Get the length of the vector.
    :param vector: np nd.array
    :return: Number, the length of the vector
    """
    # TODO: Should be replaced by np.linalg.norm!
    return np.sqrt(np.sum(np.power(vector, 2)))


def get_normalized_vector(vector):
    """
    Get the normalized form of the vector.
    :param vector: np array with two elements
    :return: np array with two elements, normalized vector
    """
    # WARN: Zero length may cause problems!
    return np.divide(vector, get_vector_length(vector))


def get_direction_vector(vector):
    """
    Get direction vector from the vector.
    :param vector: np nd.array
    :return: np array with two elements, direction vector
    """
    # TODO: Rename the function because it is misleading!
    return np.subtract(vector[1], vector[0])
