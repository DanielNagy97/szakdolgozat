import unittest

from arpt.video import Video


class VideoTest(unittest.TestCase):
    """
    Unittest for the Video class usage
    """
    
    def test_context_manager(self):
        with Video('/tmp/arpt/videos/drag.webm') as video:
            while video.get_next_frame() is not None:
                pass

    def test_frame_default_size(self):
        with Video('/tmp/arpt/videos/drag.webm') as video:
            while True:
                frame = video.get_next_frame()
                if frame is None:
                    break
                height, width, _ = frame.shape
                self.assertEqual(width, 800)
                self.assertEqual(height, 600)

    def test_frame_resizing(self):
        expected_width = 400
        expected_height = 300
        with Video('/tmp/arpt/videos/drag.webm', dimension=(expected_width, expected_height)) as video:
            while True:
                frame = video.get_next_frame()
                if frame is None:
                    break
                height, width, _ = frame.shape
                self.assertEqual(width, expected_width)
                self.assertEqual(height, expected_height)
