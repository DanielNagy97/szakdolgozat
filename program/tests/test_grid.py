import unittest

from arpt import grid
from arpt.video import Video


class GridTest(unittest.TestCase):
    """
    Unittest for the grid module
    """

    def test_calc_centers(self):
        with Video('/tmp/arpt/videos/drag.webm') as video:
            frame = video.get_next_frame()
            centers = grid.calc_centers(frame, (3, 4))
            self.assertEquals(centers.shape[0], 12)
            self.assertEquals(centers.shape[1], 2)
