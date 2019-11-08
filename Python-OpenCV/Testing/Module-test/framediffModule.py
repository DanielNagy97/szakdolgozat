import cv2

def frameDifferencing(grayFrame,oldGrayFrame,canvas):
    frameDiff = cv2.absdiff(grayFrame,oldGrayFrame)
    blur = cv2.blur(frameDiff,(20,20))
    _, thresholdedFrame = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    canvas = cv2.addWeighted(canvas,0.9,thresholdedFrame,1-0,0)
    return canvas