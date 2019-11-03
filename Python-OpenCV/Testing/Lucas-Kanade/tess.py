import cv2
import numpy

frame1 = cv2.imread("frame1.png")
frame2 = cv2.imread("frame2.png")
grayFrame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
grayFrame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

capHeight = 360
capWidth = 640

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (50,50),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.05))


cv2.namedWindow("test")

point = (149, 66)
oldPoints = numpy.array([[point[0], point[1]]], dtype=numpy.float32)

gridStep = 10

for i in range(gridStep, capWidth, gridStep):
    for j in range(gridStep, capHeight, gridStep):
        oldPoints= numpy.concatenate((oldPoints, numpy.array([[i,j]], dtype=numpy.float32)))

originalPoints = oldPoints



white = numpy.zeros([capHeight,capWidth,1],dtype=numpy.uint8)
white.fill(255)


newPoints, status, error = cv2.calcOpticalFlowPyrLK(grayFrame2, grayFrame1, oldPoints, None, **lk_params)

for k in range(int(newPoints.size/2)):
            oldX, oldY = oldPoints[k].ravel()
            newX, newY = newPoints[k].ravel()
            #cv2.circle(frame, (x,y), 5, (0, 255, 0), 4)
            cv2.arrowedLine(white, (oldX, oldY), (newX, newY), (0,0,255), 2)

while True:
    #cv2.imshow('test', frame1)
    #cv2.imshow('tes2t', frame2)
    cv2.imshow('vestorField',white)

    k = cv2.waitKey(300) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()