import cv2


class FrameBuffer(object):
    """
    Class for buffering the frames of the video stream
    """

    def __init__(self, buffer_size=10, need_grayscale=True):
        self._buffer_size = buffer_size
        self._need_grayscale = need_grayscale
        self._frames = []

    def push_frame(self, frame):
        """
        Push the frame to the buffer as the last one.
        :param frame: frame as a three dimensional NumPy array
        :return: None
        """
        assert len(self._frames) <= self._buffer_size
        if len(self._frames) == self._buffer_size:
            _ = self._frames.pop(0)
        if self._need_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._frames.append(frame)

    def __getitem__(self, index):
        """
        Provide the frames from the past with negative indices.
        For unavailable negative indices it returns with the oldest frame.
        On empty buffer it raises a ValueError exception.
        :param index: non-positive index of the frame in the buffer
        :return: the required frame
        """
        if index > 0:
            raise AttributeError(f'The index must be non-positive instead of {index}!')
        if -index < len(self._frames):
            real_index = len(self._frames) - 1 + index
            return self._frames[real_index]
        else:
            return self._frames[0]
    
    def __len__(self):
        """
        Returns the count of the frames in the buffer.
        :return: a non-negative integer of the frame count
        """
        return len(self._frames)
    
    def __str__(self):
        """
        Print some useful information about the frame buffer.
        """
        if len(self._frames) == 0:
            return 'Empty frame buffer'
        n_rows, n_columns = self._frames[0].shape
        return f'FrameBuffer, dimension: {n_columns}x{n_rows}, frame count: {len(self._frames)}'

