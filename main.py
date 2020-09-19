import numpy as np
import cv2

cap = cv2.VideoCapture("Mouth.mp4")

width =  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while (True):
    ret, frame = cap.read()
    resized = cv2.resize(frame, (width//2, height//2))
    rows, columns, channels = resized.shape
    R = cv2.getRotationMatrix2D((columns/2, rows/2), 90, 0.5)
    output = cv2.warpAffine(resized, R, (columns, rows))
    cv2.imshow('test', output)
    if (cv2.waitKey(20) & 0xFF == ord('q')):
        break


cap.release()
cv2.destroyAllWindows()
