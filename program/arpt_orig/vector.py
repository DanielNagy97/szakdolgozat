import numpy as np


def get_vector_length(vector):
    """
    Get the length of the vector.
    :param vector: np nd.array
    :return: Number, the length of the vector
    """
    return np.linalg.norm(vector)


def get_normalized_vector(vector):
    """
    Get the normalized form of the vector.
    :param vector: np array with two elements
    :return: np array with two elements, normalized vector
    """
    # WARN: Zero length may cause problems!
    return np.divide(vector, get_vector_length(vector))


def get_euclidean_vector(vector):
    """
    Get euclidean vector with starter point pushed to origin.
    :param vector: np nd.array
    :return: np array with two elements, vector
    """
    # TODO: Rename the function because it is misleading!
    return np.subtract(vector[1], vector[0])
