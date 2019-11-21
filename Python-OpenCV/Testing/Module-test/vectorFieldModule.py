import cv2
import numpy
import math
import matplotlib.pyplot as plt

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (50,50),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def caclOpticalFlow(oldGrayFrame,grayFrame,oldPoints):
    return cv2.calcOpticalFlowPyrLK(oldGrayFrame, grayFrame, oldPoints, None, **lk_params)


def vectorFieldGrid(gridStep, capWidth, capHeight):
    for i in range(gridStep, capWidth, gridStep):
        for j in range(gridStep, capHeight, gridStep):
            if i == gridStep and j == gridStep:
                oldPoints = numpy.array([[i, j]], dtype=numpy.float32)
            else:
                oldPoints = numpy.concatenate((oldPoints, numpy.array([[i,j]], dtype=numpy.float32)))
    return oldPoints

GVLforFrames=[]

def drawVectorField(canvas,oldPoints,newPoints):
    global GVLforFrames
    vectorSum = [[0.0, 0.0],[0.0, 0.0]]
    globalDirectionVector = [0.0, 0.0]
    canvas.fill(255)
    for k in range(int(newPoints.size/2)):
        oldX, oldY = oldPoints[k].ravel()
        newX, newY = newPoints[k].ravel()
        if abs(oldX-newX) >= 2 or abs(oldY-newY) >= 2:
            vectorSum[0][0] += oldX
            vectorSum[0][1] += oldY
            vectorSum[1][0] += newX
            vectorSum[1][1] += newY
            
            cv2.arrowedLine(canvas, (oldX, oldY), (newX, newY), (0,0,255), 2)

    globalVectorCanvas = numpy.zeros([250,250,1],dtype=numpy.uint8)
    globalDirectionVector[0] = vectorSum[1][0]-vectorSum[0][0]
    globalDirectionVector[1] = vectorSum[1][1]-vectorSum[0][1]
    globalDirectionVectorLength = getVectorLength(globalDirectionVector)
    GVLforFrames.append(int(globalDirectionVectorLength))
    cv2.arrowedLine(globalVectorCanvas, (0+125, 0+125), (int(globalDirectionVector[0]/8)+125, int(globalDirectionVector[1]/8)+125), (255,255,255), 2)
    cv2.imshow("GlobalMotionVector",globalVectorCanvas)


def getVectorLength(vector):
    return  math.sqrt(math.pow(vector[0],2)+math.pow(vector[1],2))

def showResults():
    ys = GVLforFrames
    xs = list(range(len(ys)))
    plt.plot(xs,ys)
    plt.ylabel('Global motion vector length')
    plt.xlabel('Frames')
    plt.show()