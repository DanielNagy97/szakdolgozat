import cv2
import numpy
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

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

    globalDirectionVector[0] = vectorSum[1][0]-vectorSum[0][0]
    globalDirectionVector[1] = vectorSum[1][1]-vectorSum[0][1]
    globalDirectionVectorLength = getVectorLength(globalDirectionVector)
    GVLforFrames = GVLforFrames[-29:]
    GVLforFrames.append(int(globalDirectionVectorLength))
    
    showResults(GVLforFrames,globalDirectionVector)

def getVectorLength(vector):
    return  math.sqrt(math.pow(vector[0],2)+math.pow(vector[1],2))


def showResults(vectorLengths,globalDirection):

    canvas = numpy.zeros([300,700,3],dtype=numpy.uint8)
    canvas.fill(255)
    cv2.line(canvas, (600, 250), (600, 50), (0,180,0), 1)
    cv2.line(canvas, (500, 150), (700, 150), (0,180,0), 1)
    cv2.line(canvas, (15, 285), (470, 285), (0,180,0), 1)
    cv2.line(canvas, (15, 285), (15, 20), (0,180,0), 1)
    cv2.putText(canvas, 'Global Resultant Vector', (250,15), cv2.FONT_HERSHEY_PLAIN , 1, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Lenght', (0,15), cv2.FONT_HERSHEY_PLAIN , 1, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Direction', (560,40), cv2.FONT_HERSHEY_PLAIN , 1, (0,0,0), 1, cv2.LINE_AA)

    a = numpy.int32(numpy.add(numpy.multiply(numpy.divide(vectorLengths,3),-1),285))

    step = 15
    i = 1
    while(i<len(a)):
        cv2.line(canvas, (step*i,a[i-1]), (step*i+step,a[i]), (0,0,255), 2)
        i+=1

    cv2.arrowedLine(canvas, (0+600, 0+150), (int(globalDirection[0]/8)+600, int(globalDirection[1]/8)+150), (0,0,0), 2)


    cv2.imshow("Test",canvas)