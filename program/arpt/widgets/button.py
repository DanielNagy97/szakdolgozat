from arpt.widget import Widget
import numpy as np
import cv2
import time


class Button(Widget):
    """
    Button widget representation
    """
    def __init__(self, position, dimension, image, action):
        """
        Initialize new button widget.
        :param position: position of the element tuple of (x,y)
        :param dimension: dimension of the element tuple of (width, height)
        :param image: source of the image file
        """
        super().__init__(position, dimension, image)
        self.action = action
        self._center_point = np.empty((2, ), dtype=np.float32)
        self._control_position = np.empty((2, ), dtype=np.float32)
        self.about_to_push = False
        self._pushed = False
        self.lk_params = {
            "winSize": (50, 50),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS |
                         cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }

    def inspect_button(self, heat_map, grid, video):
        """
        The button's working logic
        If the heat map's blob's center is inside the button->
        A tracking point will be following the palm.
        If the point get outside of the button in a given time, nothing happens
        If it stays there, the button will be pushed!
        :param heat_map: The Heat-map object
        :param grid: The grid object
        :param video: The Video object
        """
        # NOTE: Center of hmap's rects are calc-ed elsewhere too (f.e: in grab)
        if len(heat_map.bounding_rects) == 1 and not self.about_to_push:
            x, y, w, h = heat_map.bounding_rects[0]
            center = np.array((y + h/2, x + w/2),
                              dtype=np.uint8)
            self._center_point = \
                np.uint32(grid.old_points_3D[center[0],
                                             center[1]])

        if self._center_point.any() and not self.about_to_push:
            btn_pos_x, btn_pos_y = self.position
            btn_w, btn_h = self.dimension
            if ((self._center_point[0] > btn_pos_x and
                self._center_point[1] > btn_pos_y) and
                    (self._center_point[0] < btn_pos_x + btn_w) and
                    self._center_point[1] < btn_pos_y + btn_h):
                if not heat_map.bounding_rects.any():
                    self.about_to_push = True
                    x, y = self._center_point
                    self._control_position = np.array([[x, y]],
                                                      dtype=np.float32)
                    self.push_time = time.time()
                    self._center_point = np.empty((0, ), dtype=np.float32)

        if self.about_to_push:
            new_pos, status, error = \
                cv2.calcOpticalFlowPyrLK(video.old_gray_frame,
                                         video.gray_frame,
                                         self._control_position,
                                         None,
                                         **self.lk_params)
            self._control_position = new_pos
            x, y = self._control_position.ravel()
            btn_pos_x, btn_pos_y = self.position
            btn_w, btn_h = self.dimension
            if not ((x > btn_pos_x and
                    y > btn_pos_y) and
                    (x < btn_pos_x + btn_w) and
                    y < btn_pos_y + btn_h):
                self.about_to_push = False

            # Visualizing the tracker point
            cv2.circle(video.frame, (int(x), int(y)),
                       3, (0, 255, 0), 4)

            if time.time() - self.push_time >= 0.2:
                self.about_to_push = False
                self._pushed = True
