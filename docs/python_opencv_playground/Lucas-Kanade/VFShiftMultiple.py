import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)

capHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
capWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

cv2.namedWindow("VFShift")

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (50,50),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


gridStep = int(capWidth/16/2)

for i in range(gridStep, capWidth, gridStep):
    for j in range(gridStep, capHeight, gridStep):
        if i == gridStep and j == gridStep:
            oldPoints = np.array([[i, j]], dtype=np.float32)
        else:
            oldPoints= np.concatenate((oldPoints, np.array([[i,j]], dtype=np.float32)))

originalPoints = oldPoints.copy()

_, oldFrame = cap.read()
oldFrame = cv2.flip(oldFrame, 1)
oldGrayFrame = cv2.cvtColor(oldFrame,cv2.COLOR_BGR2GRAY)

class Rectangle:
    def __init__(self, posX, posY, height, width):
        self.posX = posX
        self.posY = posY
        self.height = height
        self.width = width
    velocityX = 0.0
    velocityY = 0.0
    localVectorSum = [[0.0, 0.0],[0.0, 0.0]]
    localDirectionVector = [0.0, 0.0]

rectangles = [Rectangle(200,150,100,100),Rectangle(350,100,50,50),Rectangle(420,200,50,80)]

while True:
    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame, 1)
        grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        newPoints, status, error = cv2.calcOpticalFlowPyrLK(oldGrayFrame, grayFrame, oldPoints, None, **lk_params)

        for rect in rectangles:
            rect.localVectorSum = [[0.0, 0.0],[0.0, 0.0]]
            rect.localDirectionVector = [0.0, 0.0]

            for k in range(int(newPoints.size/2)):
                oldX, oldY = oldPoints[k].ravel()
                newX, newY = newPoints[k].ravel()
                if abs(oldX-newX) >= 3 or abs(oldY-newY) >= 3:
                    if rect.posX < oldX < rect.posX + rect.width and rect.posY < oldY < rect.posY+rect.height:
                        rect.localVectorSum[0][0] += oldX
                        rect.localVectorSum[0][1] += oldY
                        rect.localVectorSum[1][0] += newX
                        rect.localVectorSum[1][1] += newY

            rect.localDirectionVector[0] = rect.localVectorSum[1][0]-rect.localVectorSum[0][0]
            rect.localDirectionVector[1] = rect.localVectorSum[1][1]-rect.localVectorSum[0][1]

            rect.velocityX +=rect.localDirectionVector[0]*0.1
            rect.velocityY +=rect.localDirectionVector[1]*0.1

            rect.posX +=rect.velocityX
            rect.posY +=rect.velocityY

            rect.velocityX *= 0.7
            rect.velocityY *= 0.7

            if rect.posX +(rect.width/2) >= capWidth:
                rect.posX -= capWidth
            if rect.posY +(rect.height/2) >= capHeight:
                rect.posY -= capHeight
            if rect.posX +(rect.width/2) < 0:
                rect.posX += capWidth
            if rect.posY+(rect.height/2) < 0:
                rect.posY += capHeight

            cv2.rectangle(frame,(int(rect.posX),int(rect.posY)),(int(rect.posX)+int(rect.width),int(rect.posY)+rect.height),(0,255,0),3)

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