from arpt.frame_buffer import FrameBuffer


class StreamManager(object):
    """
    Class for managing the video stream and its processors
    """

    def __init__(self, video):
        """
        Initialize the stream manager with a video.
        :param video: video stream
        """
        self._video = video
        self._stream_processors = []
        self._frame_buffer = FrameBuffer()

    def add_stream_processor(self, stream_processor):
        """
        Add new stream processor the the buffer
        :param stream_processor:
        :return: None
        """
        self._stream_processors.append(stream_processor)

    def process_video(self):
        """
        Process the video stream via calling the previously set stream processors.
        :return: None
        """
        for frame in self._video:
            self._frame_buffer.push_frame(frame)
            for stream_processor in self._stream_processors:
                stream_processor.process(self._frame_buffer)
