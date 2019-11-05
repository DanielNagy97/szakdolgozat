import cv2
import numpy

cap = cv2.VideoCapture("../output.avi")
capHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
capWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

_, oldFrame = cap.read()
oldFrame = cv2.flip(oldFrame, 1)
oldGrayFrame = cv2.cvtColor(oldFrame,cv2.COLOR_BGR2GRAY)

canvas = numpy.zeros([capHeight,capWidth,1],dtype=numpy.uint8)

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        frameDiff = cv2.absdiff(grayFrame,oldGrayFrame)
        blur = cv2.blur(frameDiff,(20,20))
        ret, thresholdedFrame = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)

        canvas = cv2.addWeighted(canvas,0.9,thresholdedFrame,1-0,0)
    
        cv2.imshow('frame diff ',canvas)
        cv2.imshow('original', frame)     
        
        oldGrayFrame = grayFrame.copy()

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()