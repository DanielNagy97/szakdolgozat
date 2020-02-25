import cv2
import numpy as np


class Composition():
    """
    Composition of visual elements to the video stream
    """

    @staticmethod
    def draw_widget(widget, video):
        """
        Draw the widget.
        :param widget: a widget object
        :param video: the video stream
        :return: None
        """
        pos_x, pos_y = np.uint16(widget.position)
        width, height = np.uint16(widget.dimension)

        widget.image = cv2.resize(widget.image, (width, height),
                                  interpolation=cv2.INTER_CUBIC)

        added_image = \
            cv2.addWeighted(video.frame[pos_y:pos_y+height,
                                        pos_x:pos_x+width, :],
                            0.5, widget.image[0:height, 0:width, :],
                            1-0, 0)

        video.frame[pos_y:pos_y+height,
                    pos_x:pos_x+width, :] = added_image

    @staticmethod
    def draw_expand(expand, video):
        """
        Draw the expandeble widget.
        :param widget: an expandable widget
        :param video: the video stream
        :return: None
        """
        pos_x, pos_y = np.uint16(expand.position)
        width, height = np.uint16(expand.dimension)
        act_h = np.uint16(expand.actual_height)

        expand.image = cv2.resize(expand.image, (width, height),
                                  interpolation=cv2.INTER_CUBIC)

        added_image = \
            cv2.addWeighted(video.frame[pos_y:pos_y+act_h,
                                        pos_x:pos_x+width, :],
                            0, expand.image[height-act_h:height, 0:width, :],
                            1-0, 0)

        video.frame[pos_y:pos_y+act_h,
                    pos_x:pos_x+width, :] = added_image
