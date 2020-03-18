"""
Check the frame times (sampling frequency) of the capture devices.
"""

import cv2

video = cv2.VideoCapture('/tmp/arpt/videos/drag.webm')

fps = video.get(cv2.CAP_PROP_FPS)
print(f'fps: {fps}')

n = 0
while True:
    ret, frame = video.read()
    if ret is False:
        break
    n += 1

print(f'frame count: {n}')
print(f'video length: {n / fps} seconds')
video.release()

