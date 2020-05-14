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
    vector_lenght = get_vector_length(vector)
    if vector_lenght != 0:
        return np.divide(vector, get_vector_length(vector))
    else:
        return [0, 0]


def get_euclidean_vector(vector):
    """
    Get euclidean vector with starter point pushed to origin.
    :param vector: np nd.array
    :return: np array with two elements, vector
    """

    return np.subtract(vector[1], vector[0])
