import cv2
import numpy as np
from ffpyplayer.player import MediaPlayer

cap = cv2.VideoCapture("PetMe.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
sourcePath = "Mouth.mp4"
player = MediaPlayer(sourcePath)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("PetMe out1.mp4", fourcc, 30, (width, height))
while (True):
    ret, frame = cap.read()
    audio_frame, val = player.get_frame()
    cv2.imshow("AV", frame)
    if (cv2.waitKey(30) & 0xFF == ord('q')):
        break
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()