import dlib, cv2, sys
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from math import hypot


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
    left_eye_coordinates = np.empty((0, 2), dtype=int)
    right_eye_coordinates = np.empty((0, 2), dtype=int)
    eyes_width_arr = np.empty(1, dtype="int32")

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

            for n in shape:
                n = shape[3]
                nose_coordinates = np.append(nose_coordinates, [n], axis=0)
                # cv2.circle(img_result, center=tuple((n / SCALE_FACTOR).astype(int)), radius=5, color=(0, 0, 255),
                #            thickness=-1, lineType=cv2.LINE_AA)
            for l in shape:
                l = shape[5]
                left_eye_coordinates = np.append(left_eye_coordinates, [l], axis=0)
                # cv2.circle(img_result, center=tuple((l / SCALE_FACTOR).astype(int)), radius=5, color=(255, 0, 0),
                #            thickness=-1, lineType=cv2.LINE_AA)
            for r in shape:
                r = shape[2]
                right_eye_coordinates = np.append(right_eye_coordinates, [r], axis=0)
                # cv2.circle(img_result, center=tuple((r / SCALE_FACTOR).astype(int)), radius=5, color=(0, 255, 0),
                #            thickness=-1, lineType=cv2.LINE_AA)

        eyes_width = int(hypot(int(l[0]-r[0]), int(l[1]-r[1])))
        # print(eyes_width)
        eyes_width_arr = np.append(eyes_width_arr, [eyes_width], axis=0)
        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGRA2BGR)
        cv2.imshow('result', img_result)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    eyes_width_arr_2 = eyes_width_arr[1:]
    return nose_coordinates*5, eyes_width_arr_2*5


coordinates, eyes_hypot = find_dog_face("Dog small.mp4")

cap = cv2.VideoCapture("Dog small.mp4")

print(eyes_hypot)
coordinates_x = coordinates[:, 0]
coordinates_y = coordinates[:, 1]
coordinates_smooth_x = np.empty(1, dtype="int32")
coordinates_smooth_y = np.empty(1, dtype="int32")

x_smooth_float = savgol_filter(coordinates_x, 51, 3, mode="nearest")
for x in x_smooth_float:
    x = int(x)
    coordinates_smooth_x = np.append(coordinates_smooth_x, [x], axis=0)
y_smooth_float = savgol_filter(coordinates_y, 51, 3, mode="nearest")
for y in y_smooth_float:
    y = int(y)
    coordinates_smooth_y = np.append(coordinates_smooth_y, [y], axis=0)
    # coordinates_smooth_y = np.delete(coordinates_smooth_y, [0])
# print(coordinates_smooth_x[1:])
# print(coordinates_smooth_y[1:])

# plt.plot(y_smooth, color="red")
# plt.plot(x_smooth, color="green")
# plt.plot(coordinates_smooth_x[1:])
# plt.plot(coordinates_smooth_y[1:])
# plt.show()
