import cv2
import numpy as np
import dlib
from ffpyplayer.player import MediaPlayer

cap = cv2.VideoCapture("Mouth.mp4")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
sourcePath = "Mouth.mp4"
player = MediaPlayer(sourcePath)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("Mouth_crop.mp4", fourcc, 30, (width, height))

# width2 = width + 100
# height2 = height + 100

while (True):
    ret, frame = cap.read()
    audio_frame, val = player.get_frame()
    # frame_resized = cv2.resize(frame, (width // 2, height // 2))
    # rows, columns, channels = frame.shape

    R = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 0.5)
    frame_rot = cv2.warpAffine(frame, R, (width, height))
    gray = cv2.cvtColor(frame_rot, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # print(x1, x2, y1, y2)

        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,10), 5)

        landmarks = predictor(gray, face)
        landmark_points = []
        for n in range(48, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_points.append((x, y))
        # top_left_x = landmarks.part(48).x
        # top_left_y = landmarks.part(50).y
        # bot_left_x = landmarks.part(54).x
        # bot_left_y = landmarks.part(57).y
        # cv2.rectangle(frame_rot, (top_left_x, top_left_y), (bot_left_x, bot_left_y), (255,0,10), 5)

        points = np.array(landmark_points, np.int32)
        convexhull = cv2.convexHull(points)


        # cv2.polylines(frame, [convexhull], True, (0, 255, 0), 3)
        mouth_fill = cv2.fillConvexPoly(mask, convexhull, 255)
        kernel = np.ones((10, 10), np.float32)/100
        dst = cv2.filter2D(mouth_fill, -1, kernel)
        mask = cv2.dilate(mouth_fill, kernel, dst)
        mask = cv2.blur(mouth_fill, (10, 10))

    mouth_mask = cv2.bitwise_and(frame_rot, frame_rot, mask=mask)
    cv2.imshow("mouth mask", mouth_mask)
    if (cv2.waitKey(18) & 0xFF == ord('q')):
        break
    out.write(mouth_mask)

cap.release()
out.release()
cv2.destroyAllWindows()
