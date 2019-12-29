import cv2
import numpy as np

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

gridStep = int(capWidth/16)

oldPoints = np.empty((0,2),dtype=np.float32)

for i in range(gridStep, capHeight, gridStep):
    for j in range(gridStep, capWidth, gridStep):
        oldPoints = np.append(oldPoints,
                            np.array([[j,i]], dtype=np.float32),
                            axis=0)


_, oldFrame = cap.read()
oldFrame = cv2.flip(oldFrame, 1)
oldGrayFrame = cv2.cvtColor(oldFrame,cv2.COLOR_BGR2GRAY)

rectX = 200
rectY = 150
rectW = 100
rectH = 100

rectXV = 0.0
rectYV = 0.0

oldPoints_3D = oldPoints.reshape(8,15,2)

while True:
    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame, 1)
        grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        newPoints, status, error = cv2.calcOpticalFlowPyrLK(oldGrayFrame, grayFrame, oldPoints, None, **lk_params)

        newPoints_3D = newPoints.reshape(8,15,2)

        x = np.uint8(np.floor(rectX/gridStep))
        y = np.uint8(np.floor(rectY/gridStep))
        w = np.uint8(np.floor(rectW/gridStep))
        h = np.uint8(np.floor(rectH/gridStep))

        localVectorSum = np.array([ oldPoints_3D[y:y+h,x:x+w].sum(axis=0),
                                    newPoints_3D[y:y+h,x:x+w].sum(axis=0)],
                                    dtype=np.float32).sum(axis=1)

        localDirectionVector = np.subtract(localVectorSum[1], localVectorSum[0])

        vectorCount = len(localVectorSum)

        rectXV += localDirectionVector[0]/vectorCount*0.5
        rectYV += localDirectionVector[1]/vectorCount*0.5

        rectX += rectXV
        rectY += rectYV

        rectXV *= 0.8
        rectYV *= 0.8

        if rectX+rectW >= capWidth:
            rectX = capWidth-rectW
        if rectY+rectH >= capHeight:
            rectY = capHeight-rectH
        if rectX < 0:
            rectX = 0
        if rectY < 0:
            rectY = 0

        cv2.rectangle(frame,(int(rectX),int(rectY)),(int(rectX)+int(rectW),int(rectY)+rectH),(0,255,0),3)
        cv2.arrowedLine(frame, (int(rectX+rectW/2), int(rectY+rectH/2)), (int(localDirectionVector[0]+rectX+rectW/2), int(localDirectionVector[1]+rectY+rectH/2)), (0,255,255), 2)


        oldGrayFrame = grayFrame.copy()

        cv2.imshow('VFShift', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()