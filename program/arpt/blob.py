class Blob(object):
    """
    Represents the group of adjacent pixels.
    """
    
    def __init__(self, center, size):
        self._center = center
        self._size = size

    @property
    def center(self):
        return self._center

    @property
    def size(self):
        return self._size
