import cv2
import numpy

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)

capHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
capWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


cv2.namedWindow("test")
cv2.namedWindow("vectorField")

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

whiteCanvas = numpy.zeros([capHeight,capWidth,1],dtype=numpy.uint8)
whiteCanvas.fill(255)

rectX = 200
rectY = 150
rectW = 50
rectH = 50

def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
    px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
    py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
    return int(px), int(py)

while True:
    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame, 1)
        grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        

        newPoints, status, error = cv2.calcOpticalFlowPyrLK(oldGrayFrame, grayFrame, oldPoints, None, **lk_params)

        whiteCanvas.fill(255)

        for k in range(int(newPoints.size/2)):
            oldX, oldY = oldPoints[k].ravel()
            newX, newY = newPoints[k].ravel()
            if abs(oldX-newX) >= 2 or abs(oldY-newY) >= 2:
                cv2.arrowedLine(whiteCanvas, (oldX, oldY), (newX, newY), (0,0,255), 2)
                
                if rectX < oldX < rectX+rectW and rectY < oldY < rectY+rectH:
                    #print("a regi benne van")
                    continue
                else:
                    if rectX < newX < rectX+rectW and rectY < newY < rectY+rectH:
                        #print("benne")

                        if oldX < rectX and rectY < oldY < rectY+rectH:
                            #print("1")
                            interx, intery = findIntersection(oldX,oldY,newX,newY,rectX,rectY,rectX,rectY+rectH)
                        if rectX < oldX < rectX+rectW and oldY < rectY:
                            #print("2")
                            interx, intery = findIntersection(oldX,oldY,newX,newY,rectX,rectY,rectX+rectW,rectY)
                        if  rectX < oldX < rectX+rectW and oldY > rectY+rectH:
                            interx, intery = findIntersection(oldX,oldY,newX,newY,rectX,rectY+rectH,rectX+rectW,rectY+rectH)
                            #print("3")
                        if oldX > rectX+rectW and rectY < oldY < rectY+rectH:
                            interx, intery = findIntersection(oldX,oldY,newX,newY,rectX+rectW,rectY,rectX+rectW,rectY+rectH)
                            #print("4")

                        if oldX < rectX and oldY < rectY:
                            interx, intery = findIntersection(oldX,oldY,newX,newY,0,rectY,rectX,rectY)
                            if interx < rectX:
                                interx, intery = findIntersection(interx,intery,newX,newY,rectX,rectY,rectX,rectY+rectH)
                        if oldX < rectX and oldY > rectY+rectH:
                            interx, intery = findIntersection(oldX,oldY,newX,newY,0,rectY+rectH,rectX,rectY+rectH)
                            if interx < rectX:
                                interx, intery = findIntersection(interx,intery,newX,newY,rectX,rectY,rectX,rectY+rectH)
                        if oldX > rectX+rectW and oldY < rectY:
                            interx, intery = findIntersection(oldX,oldY,newX,newY,capWidth,rectY,rectX+rectW,rectY)
                            if interx > rectX+rectW:
                                interx, intery = findIntersection(interx,intery,newX,newY,rectX+rectW,rectY,rectX+rectW,rectY+rectY)
                        if oldX > rectX+rectW and oldY > rectY+rectY:
                            interx, intery = findIntersection(oldX,oldY,newX,newY,capWidth,rectY+rectY,rectX+rectW,rectY+rectY)
                            if interx > rectX+rectW:
                                interx, intery = findIntersection(interx,intery,newX,newY,rectX+rectW,rectY,rectX+rectW,rectY+rectY)

                        try:
                            dvX = newX-interx
                            dvY = newY-intery
                            rectX = rectX+dvX
                            rectY = rectY+dvY
                        except:
                            pass
                    else:
                        #print("nincs benne")
                        continue

        cv2.rectangle(frame,(int(rectX),int(rectY)),(int(rectX)+int(rectW),int(rectY)+rectH),(0,255,0),3)
        oldPoints = originalPoints.copy()
        oldGrayFrame = grayFrame.copy()

        cv2.imshow('test', frame)
        cv2.imshow('vectorField',whiteCanvas)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()