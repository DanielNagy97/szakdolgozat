import cv2
import numpy

# Video source from webcam
cap = cv2.VideoCapture(0)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (50,50),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cv2.namedWindow("test")

rectX = 200
rectY = 150
rectW = 100
rectH = 100

oldPoints = numpy.array([[rectX, rectY]], dtype=numpy.float32)

_, oldFrame = cap.read()
oldFrame = cv2.flip(oldFrame, 1)
oldGrayFrame = cv2.cvtColor(oldFrame,cv2.COLOR_BGR2GRAY)

grabbed = False

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        if grabbed:
            newPoints, status, error = cv2.calcOpticalFlowPyrLK(oldGrayFrame, grayFrame, oldPoints, None, **lk_params)
            oldPoints = newPoints
            x, y = newPoints.ravel()

        else:
            x, y = oldPoints.ravel()
        
        rectX = x
        rectY = y
        
        cv2.circle(frame, (x,y), 5, (0, 0, 255), 2)
        cv2.rectangle(frame,(int(rectX),int(rectY)),(int(rectX)+int(rectW),int(rectY)+rectH),(0,255,0),3)

        oldGrayFrame = grayFrame.copy()
        cv2.imshow('test', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        #press space to grab/release
        if k == 32:
            grabbed = not grabbed
    else:
        break

cap.release()
cv2.destroyAllWindows()