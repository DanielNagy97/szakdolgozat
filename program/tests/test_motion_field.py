import unittest

from arpt.frame_buffer import FrameBuffer
from arpt import motion_field
from arpt.video import Video


class MotionFieldTest(unittest.TestCase):
    """
    Unittest for the motion_field module
    """

    def test_calc_optical_flow(self):
        frames = FrameBuffer()
        with Video('/tmp/arpt/videos/drag.webm') as video:
            frames.push_frame(video.get_next_frame())
            frames.push_frame(video.get_next_frame())
        motion_vectors = motion_field.calc_optical_flow(frames[-1], frames[0], (3, 4))
        self.assertEquals(motion_vectors.shape[0], 3)
        self.assertEquals(motion_vectors.shape[1], 4)
        self.assertEquals(motion_vectors.shape[2], 2)
