import cv2
import numpy
import vectorFieldModule as vf
import framediffModule as fd
import initModule as init

#windows
init.init()

#capture
cap = cv2.VideoCapture("vtest.avi")

#setting capture device
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)

#getting frame dimensions
capHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
capWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

#generating points for vector field
gridStep = int(capWidth/16)
oldPoints = vf.vectorFieldGrid(gridStep,capWidth,capHeight)
originalPoints = oldPoints.copy()

#reading the very first frame
_, oldFrame = cap.read()
oldFrame = cv2.flip(oldFrame, 1)
oldGrayFrame = cv2.cvtColor(oldFrame,cv2.COLOR_BGR2GRAY)

#initializing canvases for drawing
vectorFieldCanvas = numpy.zeros([capHeight,capWidth,1],dtype=numpy.uint8)
vectorFieldCanvas.fill(255)
frameDiffCanvas = numpy.zeros([capHeight,capWidth,1],dtype=numpy.uint8)

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        newPoints, status, error = vf.caclOpticalFlow(oldGrayFrame,grayFrame,oldPoints)

        vf.drawVectorField(vectorFieldCanvas,oldPoints,newPoints)

        frameDiffCanvas = fd.frameDifferencing(grayFrame,oldGrayFrame,frameDiffCanvas)

        oldPoints = originalPoints.copy()
        oldGrayFrame = grayFrame.copy()

        cv2.imshow('test', frame)
        cv2.imshow('vectorField',vectorFieldCanvas)
        cv2.imshow('frameDiff',frameDiffCanvas)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()