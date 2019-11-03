import cv2
import numpy

cap = cv2.VideoCapture(0)
capHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
capWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ret, current_frame = cap.read()
previous_frame = current_frame

canvas = numpy.zeros([capHeight,capWidth,1],dtype=numpy.uint8)

while(cap.isOpened()):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    

    frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray)
    blur = cv2.blur(frame_diff,(20,20))
    #frame = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    ret, frame = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)


    canvas = cv2.addWeighted(canvas,0.95,frame,1-0,0)
    
    cv2.imshow('frame diff ',canvas)
    cv2.imshow('original', current_frame)     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    previous_frame = current_frame.copy()
    ret, current_frame = cap.read()

cap.release()
cv2.destroyAllWindows()