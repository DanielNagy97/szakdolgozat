
class frame_diff():
    def __init__(self):
        pass

    def frame_differencing(self, gray_frame,old_gray_frame,canvas):
        frame_diff = cv2.absdiff(gray_frame,old_gray_frame)
        blur = cv2.blur(frame_diff,(20,20))
        _, thresholded_frame = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
        canvas = cv2.addWeighted(canvas,0.9,thresholded_frame,1-0,0)
        return canvas
