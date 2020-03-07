import cv2
import numpy as np

im = cv2.imread("Nelly.png")

hh = 0
stacked = im

for i in range(7):
    im_pyrD = cv2.pyrDown(im)
    h, w, c = im_pyrD.shape
    white_border = np.zeros([int(h/2) + hh, w, c], dtype=np.uint8)
    white_border.fill(255)
    im_w_border = np.vstack((white_border, im_pyrD))
    im_w_border = np.vstack((im_w_border, white_border))
    stacked = np.hstack((stacked, im_w_border))

    im = im_pyrD
    hh = hh + int(h/2)

cv2.imshow("Gaussian_Pyramid", stacked)
cv2.waitKey()

cv2.imwrite("pyramid.png", stacked)
