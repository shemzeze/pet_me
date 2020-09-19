import cv2
import numpy as np
from ffpyplayer.player import MediaPlayer

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture("Dog.mp4")
img = cv2.imread("cam_guidelines.png")
sourcePath = "Dog.mp4"
player = MediaPlayer(sourcePath)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("dog_out_1.mp4", fourcc, 30, (width, height))


while True:
    ret, frame = cap.read()
    ret, frame2 = cap2.read()
    audio_frame, val = player.get_frame()
    # frame2_resized = cv2.resize(frame2, (500, 500))
    frame2_resized = cv2.resize(frame2, (width2 // 2, height2 // 2))
    rows, columns, channels = frame2_resized.shape
    print(frame2_resized)
    R = cv2.getRotationMatrix2D((columns / 2, rows / 2), 270, 0.5)
    frame2_rot = cv2.warpAffine(frame2_resized, R, (columns, rows))
    img_resize = cv2.resize(img, (width, height))
    together = cv2.addWeighted(img_resize, 0.25, frame, 1, 0, frame)
    cv2.imshow('test', frame2_rot)
    cv2.imshow("bvid", together)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
    # out.write(frame)
    # out.write(frame2_rot)

    # h_frame, w_frame, c_frame = frame.shape
    # center_y = int(h_frame / 2)
    # center_x = int(w_frame / 2)
    # img_resize = cv2.resize(img, (width, height))
    # together = cv2.addWeighted(img_resize, 0.25, frame, 1, 0, frame)
    # togeth = cv2.add(frame, img_resize)


cap.release()
cv2.destroyAllWindows()