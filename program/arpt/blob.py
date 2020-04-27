class Blob(object):
    """
    Represents the group of adjacent pixels.
    """
    
    def __init__(self, center, size):
        """
        Initialize the blob.
        :param center: center of the blob as (row, column)
        :param size: the estimated size of the blob as a float value
        """
        self._center = center
        self._size = size

    @property
    def center(self):
        return self._center

    @property
    def size(self):
        return self._size

