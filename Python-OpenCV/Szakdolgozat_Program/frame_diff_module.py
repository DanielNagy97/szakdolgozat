import cv2

def frame_differencing(gray_frame,old_gray_frame,canvas):
    frame_diff = cv2.absdiff(gray_frame,old_gray_frame)
    blur = cv2.blur(frame_diff,(20,20))
    _, thresholded_frame = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    canvas = cv2.addWeighted(canvas,0.9,thresholded_frame,1-0,0)
    return canvas