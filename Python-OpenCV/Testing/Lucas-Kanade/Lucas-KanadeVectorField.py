import cv2
import numpy

# Video source from webcam
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("vtest.avi")
capHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
capWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (30,30),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


cv2.namedWindow("test")

point = (149, 66)
oldPoints = numpy.array([[point[0], point[1]]], dtype=numpy.float32)


#oldPoints= numpy.concatenate((oldPoints, numpy.array([[300,300]], dtype=numpy.float32)))

gridStep = 20

for i in range(gridStep, capWidth, gridStep):
    for j in range(gridStep, capHeight, gridStep):
        oldPoints= numpy.concatenate((oldPoints, numpy.array([[i,j]], dtype=numpy.float32)))

originalPoints = oldPoints

_, oldFrame = cap.read()
oldFrame = cv2.flip(oldFrame, 1)
oldGrayFrame = cv2.cvtColor(oldFrame,cv2.COLOR_BGR2GRAY)


while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        newPoints, status, error = cv2.calcOpticalFlowPyrLK(oldGrayFrame, grayFrame, oldPoints, None, **lk_params)

        for k in range(int(newPoints.size/2)):
            oldX, oldY = oldPoints[k].ravel()
            newX, newY = newPoints[k].ravel()

            #cv2.circle(frame, (x,y), 5, (0, 255, 0), 4)
            cv2.arrowedLine(frame, (oldX, oldY), (newX, newY), (0,0,255), 2)

        
        oldPoints = originalPoints.copy()
        oldGrayFrame = grayFrame.copy()
        cv2.imshow('test', frame)

        k = cv2.waitKey(100) & 0xFF
        if k == 27:
            break
        if k == 32:
            oldPoints = originalPoints

    else:
        break

cap.release()
cv2.destroyAllWindows()