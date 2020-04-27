import numpy as np
import cv2

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
        ret, thresholded_heat = cv2.threshold(np.uint8(heats), 40, 255,
                                              cv2.ADAPTIVE_THRESH_MEAN_C)
        contours, hierarchy = cv2.findContours(thresholded_heat,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            if rect_area >= min_area:
                center = (y + h / 2, x + w / 2)
                blob = Blob(center, rect_area)
                blobs.append(blob)
        return blobs

    @property
    def blobs(self):
        return self._blobs

