import dlib, cv2, sys
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


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
                nose_coordinates = np.append(nose_coordinates, [p], axis=0)


        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGRA2BGR)
        cv2.imshow('result', img_result)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    return nose_coordinates*5


# print(find_dog_face("Dog small.mp4"))
coordinates = find_dog_face("Dog small.mp4")
coordinates_x = coordinates[:, 0]
coordinates_y = coordinates[:, 1]
coordinates_smooth_x = np.empty(1, dtype="int32")
coordinates_smooth_y = np.empty(1, dtype="int32")

x_smooth_float = savgol_filter(coordinates_x, 51, 3, mode="nearest")
for x in x_smooth_float:
    x = int(x)
    coordinates_smooth_x = np.append(coordinates_smooth_x, [x], axis=0)
    # coordinates_smooth_x = np.delete(coordinates_smooth_x, [0])

    # coordinates_smooth_x.reshape((1, 1))
# print(coordinates_smooth_x)
y_smooth_float = savgol_filter(coordinates_y, 51, 3, mode="nearest")
for y in y_smooth_float:
    y = int(y)
    coordinates_smooth_y = np.append(coordinates_smooth_y, [y], axis=0)
    # coordinates_smooth_y = np.delete(coordinates_smooth_y, [0])
print(coordinates_smooth_x[1:])
print(coordinates_smooth_y[1:])
# coordinates_smooth = np.empty(2, dtype=int)
# coordinates_smooth = np.append(coordinates_smooth, [coordinates_smooth_x], axis=0)
# print(coordinates_smooth)
# plt.plot(y_smooth, color="red")
# plt.plot(x_smooth, color="green")
plt.plot(coordinates_smooth_x[1:])
plt.plot(coordinates_smooth_y[1:])
plt.show()
