import numpy as np
import cv2

cap = cv2.VideoCapture(0)

MOG2 = False

if MOG2:
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
else:
    backSub = cv2.createBackgroundSubtractorKNN()

while(True):
    ret, frame = cap.read()

    fgmask = backSub.apply(frame)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()