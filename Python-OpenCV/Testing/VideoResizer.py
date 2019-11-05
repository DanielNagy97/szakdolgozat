
import cv2
import numpy as np
 
cap = cv2.VideoCapture('testFootage.mp4')
 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc,5,(640,360))
 
while True:
    ret, frame = cap.read()
    if ret:
        b = cv2.resize(frame,(640,360),fx=0,fy=0, interpolation = cv2.INTER_LINEAR)
        out.write(b)

        cv2.imshow('test', b)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break
    else:
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()