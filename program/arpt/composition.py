import cv2
import numpy as np


class Composition():
    """
    Composition of visual elements to the video stream
    """

    @staticmethod
    def draw_widget(widget, video, transparency):
        """
        Draw the widget.
        :param widget: a widget object
        :param video: the video stream
        :return: None
        """
        pos_x, pos_y = np.uint16(widget.position)
        width, height = np.uint16(widget.dimension)

        cropped_frame = video.frame[pos_y:pos_y+height,
                                    pos_x:pos_x+width, :]

        if widget.image.shape[2] == 4 and widget.transparent:

            overlay_img = widget.image[:, :, :3]   # BRG planes
            overlay_mask = widget.image[:, :, 3:]  # alpha plane

            if transparency != 0:
                overlay_mask = np.uint8(np.multiply(overlay_mask,
                                                    1-transparency))

            # Inverse mask
            background_mask = np.subtract(255, overlay_mask)

            # Turn the masks into three channel
            # to use them as weights
            overlay_mask = cv2.cvtColor(overlay_mask,
                                        cv2.COLOR_GRAY2BGR)
            background_mask = cv2.cvtColor(background_mask,
                                           cv2.COLOR_GRAY2BGR)

            # Convert the images to floating point in range 0.0 - 1.0
            background_part =\
                np.multiply((np.multiply(cropped_frame,
                                         (1 / 255.0))),
                            (np.multiply(background_mask,
                                         (1 / 255.0))))
            overlay_part =\
                np.multiply((np.multiply(overlay_img,
                                         (1 / 255.0))),
                            (np.multiply(overlay_mask,
                                         (1 / 255.0))))

            # Finally just add them together
            blended = np.uint8(cv2.addWeighted(background_part, 255.0,
                                               overlay_part, 255.0, 0))

        elif transparency != 0:
            # If the widget image is missing the alpha channel
            # Or it is viewed as not transparent
            blended = cv2.addWeighted(cropped_frame, transparency,
                                      widget.image[0:height, 0:width, :3],
                                      1-transparency, 0)
        else:
            blended = widget.image[0:height, 0:width, :3]

        video.frame[pos_y:pos_y+height,
                    pos_x:pos_x+width, :] = blended
