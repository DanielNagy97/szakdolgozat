import cv2

cap = cv2.VideoCapture(0)
ret, current_frame = cap.read()
previous_frame = current_frame

while(cap.isOpened()):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    

    frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray)
    blur = cv2.blur(frame_diff,(20,20))
    #frame = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    ret, frame = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    cv2.imshow('frame diff ',frame)      
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    previous_frame = current_frame.copy()
    ret, current_frame = cap.read()

cap.release()
cv2.destroyAllWindows()