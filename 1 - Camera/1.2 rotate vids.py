import cv2
import numpy as np
from ffpyplayer.player import MediaPlayer

cap = cv2.VideoCapture("Dog.mp4")
sourcePath = "Dog.mp4"
player = MediaPlayer(sourcePath)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("dog_out_1.mp4", fourcc, 30, (width, height))


while True:
    ret, frame = cap.read()
    audio_frame, val = player.get_frame()
    # frame2_resized = cv2.resize(frame2, (500, 500))
    frame_resized = cv2.resize(frame, (width // 2, height // 2))
    rows, columns, channels = frame_resized.shape
    R = cv2.getRotationMatrix2D((columns / 2, rows / 2), 270, 0.5)
    frame2_rot = cv2.warpAffine(frame_resized, R, (columns, rows))
    # img_resize = cv2.resize(img, (width, height))
    # together = cv2.addWeighted(img_resize, 0.25, frame, 1, 0, frame)
    # cv2.imshow('test', frame)
    cv2.imshow("bvid", frame2_rot)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
    # out.write(frame)
    out.write(frame2_rot)


cap.release()
cv2.destroyAllWindows()