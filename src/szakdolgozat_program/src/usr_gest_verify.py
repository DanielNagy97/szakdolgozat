import os
import numpy as np
import cv2


class NestedGesture(object):
    """
    Class representation of nested OCR-gestures
    """
    def __init__(self, image):
        """
        Initializing with stater gesture image
        :param image: The OCR-gesture image
        """
        self._image = image

    def update_nested_image(self, image_to_nest):
        """
        Updating the nested image
        :param image_to_nest: Ocr-gesture image for nesting
        """
        self._image = np.minimum(self._image, image_to_nest)

    def calc_black_pixels(self):
        """
        Counting the black pixels on the nested image
        """
        self._n_black_pix = np.sum(self._image == 0)
        print('Number of black pixels:', self._n_black_pix)

    def calc_intensity_of_rows_cols(self):
        """
        Counting the black pixels by rows and collumns
        """
        self._rows_intensity_count = np.sum(self._image == 0, axis=1)
        self._cols_intensity_count = np.sum(self._image == 0, axis=0)
        print("Intensity of the rows:", self._rows_intensity_count)
        print("Intensity of the cols:", self._cols_intensity_count)

    def calc_center_of_mass(self):
        """
        Calculating the center of the mass (index)
        """
        h, w = self._image.shape
        weighted_row_average = np.average(range(1, h+1),
                                          weights=self._rows_intensity_count)
        weighted_col_average = np.average(range(1, w+1),
                                          weights=self._cols_intensity_count)
        a = np.uint8(np.around(weighted_col_average)-1)
        b = np.uint8(np.around(weighted_row_average)-1)
        self._center_of_mass_index = (a, b)
        print("Index of center of mass:", self._center_of_mass_index)

    @property
    def image(self):
        return self._image

    @property
    def n_black_pix(self):
        return self._n_black_pix

    @property
    def rows_intensity_count(self):
        return self._rows_intensity_count

    @property
    def cols_intensity_count(self):
        return self._cols_intensity_count

    @property
    def center_of_mass_index(self):
        return self._center_of_mass_index


def read_gestures_from_file(source):
    """
    Reading gestures from file and making NestedGesture objects
    :param source: The source of the root folder of user-gestures
    :return: List of NestedGesture objects
    """
    nested_gesture_list = []
    usr_gest_list = os.listdir(source)

    for usr_gest in usr_gest_list:
        files = os.listdir(source+usr_gest)
        im = cv2.imread(source+usr_gest+"/"+files[0], 0)
        vars()[usr_gest] = NestedGesture(im)

        for i in range(len(files)):
            im = cv2.imread(source+usr_gest+"/"+files[i], 0)
            vars()[usr_gest].update_nested_image(im)
        nested_gesture_list.append(vars()[usr_gest])
    return nested_gesture_list


def analize_nested_gestures(nested_gesture_list):
    """
    Analising NestedGestures by differenct aspects
    :param nested_gesture_list: List of NestedGesture objects
    """
    for nested_gest in nested_gesture_list:
        nested_gest.calc_black_pixels()
        nested_gest.calc_intensity_of_rows_cols()
        nested_gest.calc_center_of_mass()

        cv2.imshow("Nested gesture", nested_gest.image)
        cv2.waitKey()
        print("----------")


if __name__ == "__main__":
    nested_gest_list = read_gestures_from_file("./usr_gest/")
    analize_nested_gestures(nested_gest_list)
