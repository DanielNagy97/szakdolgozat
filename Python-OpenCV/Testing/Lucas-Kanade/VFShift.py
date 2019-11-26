import cv2
import numpy

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)

capHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
capWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

#cv2.namedWindow("VFShift")
cv2.namedWindow("VFShift", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("VFShift",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (50,50),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


gridStep = int(capWidth/16/2)

for i in range(gridStep, capWidth, gridStep):
    for j in range(gridStep, capHeight, gridStep):
        if i == gridStep and j == gridStep:
            oldPoints = numpy.array([[i, j]], dtype=numpy.float32)
        else:
            oldPoints= numpy.concatenate((oldPoints, numpy.array([[i,j]], dtype=numpy.float32)))

originalPoints = oldPoints.copy()

_, oldFrame = cap.read()
oldFrame = cv2.flip(oldFrame, 1)
oldGrayFrame = cv2.cvtColor(oldFrame,cv2.COLOR_BGR2GRAY)

rectX = 200
rectY = 150
rectW = 100
rectH = 100

rectXV = 0.0
rectYV = 0.0

while True:
    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame, 1)
        grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        newPoints, status, error = cv2.calcOpticalFlowPyrLK(oldGrayFrame, grayFrame, oldPoints, None, **lk_params)

        localVectorSum = [[0.0, 0.0],[0.0, 0.0]]
        localDirectionVector = [0.0, 0.0]

        for k in range(int(newPoints.size/2)):
            oldX, oldY = oldPoints[k].ravel()
            newX, newY = newPoints[k].ravel()
            if abs(oldX-newX) >= 2 or abs(oldY-newY) >= 2:
                if rectX < oldX < rectX+rectW and rectY < oldY < rectY+rectH:
                    localVectorSum[0][0] += oldX
                    localVectorSum[0][1] += oldY
                    localVectorSum[1][0] += newX
                    localVectorSum[1][1] += newY
                    cv2.arrowedLine(frame, (oldX, oldY), (newX, newY), (0,0,255), 2)

        localDirectionVector[0] = localVectorSum[1][0]-localVectorSum[0][0]
        localDirectionVector[1] = localVectorSum[1][1]-localVectorSum[0][1]

        rectXV +=localDirectionVector[0]*0.1
        rectYV +=localDirectionVector[1]*0.1

        rectX +=rectXV
        rectY +=rectYV

        rectXV *= 0.7
        rectYV *= 0.7

        if rectX+(rectW/2) >= capWidth:
            rectX -= capWidth
        if rectY+(rectH/2) >= capHeight:
            rectY -= capHeight
        if rectX+(rectW/2) < 0:
            rectX += capWidth
        if rectY+(rectH/2) < 0:
            rectY += capHeight

        cv2.rectangle(frame,(int(rectX),int(rectY)),(int(rectX)+int(rectW),int(rectY)+rectH),(0,255,0),3)
        cv2.arrowedLine(frame, (int(rectX+rectW/2), int(rectY+rectH/2)), (int(localDirectionVector[0]+rectX+rectW/2), int(localDirectionVector[1]+rectY+rectH/2)), (0,255,255), 2)
        oldPoints = originalPoints.copy()
        oldGrayFrame = grayFrame.copy()

        cv2.imshow('VFShift', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()