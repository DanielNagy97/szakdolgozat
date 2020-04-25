from arpt.blob import Blob


class HeatMap(object):
    """
    Represents the heatmap as the set regions with higher motion rates.
    """
    
    def __init__(self, lengths, sensitivity=10, min_area=2):
        """
        Initialize the Heat Map
        :param grid: the grid object
        :param sensitivity: sensitivity value for displaying motion vector lengths
        :param min_area: minimum area of motion blob to analyse
        """
        heats = self.calc_heats(lengths, sensitivity)
        self._blobs = self.find_blobs(heats, min_area)

    @staticmethod
    def calc_heats(lengths, sensitivity):
        heats = np.int32(np.multiply(lengths, sensitivity))
        heats = np.where(heats > 255, 255, heats)
        return heats

    @staticmethod
    def find_blobs(heats, min_area):
        pass