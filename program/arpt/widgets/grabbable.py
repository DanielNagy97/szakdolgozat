from arpt.widget import Widget


class Grabbable(Widget):
    """
    Grabbable widget representation
    """
    def __init__(self, position, dimension, image, transparent):
        """
        Initialize new Grabbable widget.
        :param position: position of the element tuple of (x,y)
        :param dimension: dimension of the element tuple of (width, height)
        :param image: source of the image file
        """
        super().__init__(position, dimension, image, transparent)
        self.grabbed = False
        self.offset_x = 0
        self.offset_y = 0

    def update_position(self, grab, video):
        """
        Updating widget's position by the grab-position
        :param grab: The Grab object
        :param video: The Video object
        """
        if grab.grabbed:
            g_pos_x, g_pos_y = self._position
            g_w, g_h = self._dimension
            x, y = grab.position.ravel()

            if not self.grabbed:
                if ((x > g_pos_x and
                        y > g_pos_y) and
                        (x < g_pos_x + g_w) and
                        y < g_pos_y + g_h):
                    self.grabbed = True
                    self.offset_x = x-g_pos_x
                    self.offset_y = y-g_pos_y
                else:
                    grab.grabbed = False
            else:
                self._position = (x-self.offset_x, y-self.offset_y)

                pos_x, pos_y = self._position
                g_w, g_h = self._dimension
                vid_w, vid_h = video.dimension

                if pos_x < 0:
                    pos_x = 0
                if pos_y < 0:
                    pos_y = 0
                if pos_x+g_w > vid_w:
                    pos_x = vid_w-g_w
                if pos_y+g_h > vid_h:
                    pos_y = vid_h-g_h

                self._position = (pos_x, pos_y)
        else:
            self.grabbed = False
