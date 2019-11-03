import cv2
import numpy

# Video source from webcam
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)

capHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
capWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (50,50),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


cv2.namedWindow("test")

point = (149, 66)
oldPoints = numpy.array([[point[0], point[1]]], dtype=numpy.float32)


#oldPoints= numpy.concatenate((oldPoints, numpy.array([[300,300]], dtype=numpy.float32)))

gridStep = 30

for i in range(gridStep, capWidth, gridStep):
    for j in range(gridStep, capHeight, gridStep):
        oldPoints= numpy.concatenate((oldPoints, numpy.array([[i,j]], dtype=numpy.float32)))

originalPoints = oldPoints

_, oldFrame = cap.read()
oldFrame = cv2.flip(oldFrame, 1)
oldGrayFrame = cv2.cvtColor(oldFrame,cv2.COLOR_BGR2GRAY)

white = numpy.zeros([capHeight,capWidth,1],dtype=numpy.uint8)
white.fill(255)

count=0

while True:
    ret, frame = cap.read()

    if ret and count == 0:
        frame = cv2.flip(frame, 1)
        grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        newPoints, status, error = cv2.calcOpticalFlowPyrLK(oldGrayFrame, grayFrame, oldPoints, None, **lk_params)


        white.fill(255)

        for k in range(int(newPoints.size/2)):
            oldX, oldY = oldPoints[k].ravel()
            newX, newY = newPoints[k].ravel()

            #cv2.circle(frame, (x,y), 5, (0, 255, 0), 4)
            cv2.arrowedLine(white, (oldX, oldY), (newX, newY), (0,0,255), 2)

        
        oldPoints = originalPoints.copy()
        oldGrayFrame = grayFrame.copy()

        cv2.imshow('test', frame)
        cv2.imshow('vestorField',white)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    if not ret:
        break

    count+=1
    if count == 1:
        count=0


cap.release()
cv2.destroyAllWindows()