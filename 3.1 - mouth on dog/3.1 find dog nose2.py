import dlib, cv2, sys
from imutils import face_utils
import numpy as np
vid = "Dog_1.mp4"


def find_dog_face(vid):
    nose_coordinates = []
    SCALE_FACTOR = 0.3
    vid = sys.argv[0]
    cap = cv2.VideoCapture("Dog_1.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')
    predictor = dlib.shape_predictor('landmarkDetector.dat')

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

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
                nose_coordinates = np.array(p)
                # print(nose_coordinates)
                # return nose_coordinates


# print(find_dog_face("Dog_1.mp4"))
find_dog_face("Dog_1.mp4")
