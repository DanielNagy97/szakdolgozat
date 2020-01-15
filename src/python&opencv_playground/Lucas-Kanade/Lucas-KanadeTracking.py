import cv2
import numpy

# Video source from webcam
cap = cv2.VideoCapture(0)

# Random image to paste
image = cv2.imread('random.jpeg', 1)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# Select one point with mouse
def selectPoint(event, x, y, flags, params):
    global point, pointSelected, oldPoints
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        pointSelected = True
        oldPoints = numpy.array([[x, y]], dtype=numpy.float32)


cv2.namedWindow("test")
cv2.setMouseCallback("test", selectPoint)
pointSelected = False
point = ()
oldPoints = numpy.array([[]])

# _, oldFrame = cap.read()
# oldFrame = cv2.flip(oldFrame, 1)
# oldGrayFrame = cv2.cvtColor(oldFrame,cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        if pointSelected:
            #cv2.circle(frame, point, 5, (0, 0, 255), 2)

            newPoints, status, error = cv2.calcOpticalFlowPyrLK(oldGrayFrame, grayFrame, oldPoints, None, **lk_params)
            oldPoints = newPoints
            x, y = newPoints.ravel()
            cv2.circle(frame, (x,y), 5, (0, 0, 255), 2)
            added_image = cv2.addWeighted(frame[int(y):int(y)+256,int(x):int(x)+256,:],0.2,image[0:256,0:256,:],1-0.2,0)
            # Change the region with the result
            frame[int(y):int(y)+256,int(x):int(x)+256,:] = added_image
        
        oldGrayFrame = grayFrame.copy()
        cv2.imshow('test', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()