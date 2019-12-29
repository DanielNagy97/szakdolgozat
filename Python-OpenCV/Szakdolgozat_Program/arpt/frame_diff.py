import cv2

class frame_diff():
    def __init__(self):
        pass

    def frame_differencing(self, video, canvas):
        self.frame_diff = cv2.absdiff(video.gray_frame, video.old_gray_frame)
        self.blur = cv2.blur(self.frame_diff, (20, 20))
        
        _, self.thresholded_frame = cv2.threshold(self.blur, 20, 255, cv2.THRESH_BINARY)
        canvas.update(cv2.addWeighted(canvas.canvas, 0.9, self.thresholded_frame, 1-0, 0))