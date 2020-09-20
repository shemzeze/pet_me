import dlib, cv2, sys
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt

vid = "Dog small.mp4"


def find_dog_face(vid):
    SCALE_FACTOR = 0.2
    vid = sys.argv[0]
    cap = cv2.VideoCapture("Dog small.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')
    predictor = dlib.shape_predictor('landmarkDetector.dat')
    nose_coordinates = np.empty((0, 2), dtype=int)
    # print(nose_coordinates)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break
        img_result = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, dsize=None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = detector(frame_gray, upsample_num_times=1)

        for i, d in enumerate(roi):

            x1, y1 = int(d.rect.left() / SCALE_FACTOR), int(d.rect.top() / SCALE_FACTOR)
            x2, y2 = int(d.rect.right() / SCALE_FACTOR), int(d.rect.bottom() / SCALE_FACTOR)
            shape = predictor(frame_gray, d.rect)
            shape = face_utils.shape_to_np(shape)

            for m, p in enumerate(shape):
                p = shape[3]
                nose_coordinates = np.append(nose_coordinates, ([p]), axis=0)
                # print(p)
        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGRA2BGR)
        cv2.imshow('result', img_result)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    return nose_coordinates//SCALE_FACTOR


# print(find_dog_face("Dog small.mp4"))
coordinates = find_dog_face(("Dog small.mp4"))
for row in coordinates:
    if row - (row-1) >= 10:
        row2 = (row + (row-1))/2
    print(row2)
