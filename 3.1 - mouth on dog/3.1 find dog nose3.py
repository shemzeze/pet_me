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
            nose = shape[3]
            # print(nose)
            nose_coordinates = np.append(nose_coordinates, [nose], axis=0)
            left_eye = shape[5]
            left_eye_coordinates = np.append(left_eye_coordinates, [left_eye], axis=0)
            right_eye = shape[2]
            right_eye_coordinates = np.append(right_eye_coordinates, [right_eye], axis=0)

        # eyes_width = int(hypot(int(left_eye[0]-right_eye[0]), int(left_eye[1]-right_eye[1])))
        # # print(eyes_width)
        # eyes_width_arr = np.append(eyes_width_arr, [eyes_width], axis=0)
        # print(n)
        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGRA2BGR)
        cv2.imshow('result', img_result)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    eyes_width_arr_2 = eyes_width_arr[1:]
    return nose_coordinates*5, eyes_width_arr_2*5


coordinates, eyes_hypot = find_dog_face("Dog small.mp4")
# print(coordinates)
coordinates_x = coordinates[:, 0]
coordinates_y = coordinates[:, 1]
coordinates_smooth_x = np.empty(1, dtype="int32")
coordinates_smooth_y = np.empty(1, dtype="int32")

x_smooth_float = savgol_filter(coordinates_x, 21, 3, mode="nearest")
for x in x_smooth_float:
    x = int(x)
    coordinates_smooth_x = np.append(coordinates_smooth_x, [x], axis=0)
y_smooth_float = savgol_filter(coordinates_y, 21, 3, mode="nearest")
for y in y_smooth_float:
    y = int(y)
    coordinates_smooth_y = np.append(coordinates_smooth_y, [y], axis=0)
# plt.plot(coordinates_x, color="red")
# plt.plot(coordinates_y, color="red")
# plt.plot(coordinates_smooth_x[1:], color="green")
# plt.plot(coordinates_smooth_y[1:], color="green")
# print(coordinates_smooth_x)
# print(coordinates_smooth_y)
# plt.show()
coordinates_smooth_final_x = coordinates_smooth_x[1:]
coordinates_smooth_final_y = coordinates_smooth_y[1:]
# print(coordinates_smooth_final_x)
# print(coordinates_smooth_final_y)
cap = cv2.VideoCapture("Dog small.mp4")
cap2 = cv2.VideoCapture("Mouth_crop.mp4")
f = 0
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("dog_out_smooth.mp4", fourcc, 30, (width, height))

while cap.isOpened():
    ret, final_frame = cap.read()
    ret, frame2 = cap2.read()
    if not ret:
        break
    frame2_alpha = cv2.cvtColor(frame2, cv2.COLOR_BGR2BGRA)
    frame2_resize = cv2.resize(frame2_alpha, (width2 // 3, height2 // 3))
    # print(frame2_resize.shape)
    frame2_cropped = frame2_resize[140:280, 250:390]
    frame2_gray = cv2.cvtColor(frame2_cropped, cv2.COLOR_BGR2GRAY)
    _, mouth_mask = cv2.threshold(frame2_gray, 5, 255, cv2.THRESH_BINARY_INV)
    frame2_darker = cv2.addWeighted(frame2_cropped, 1, np.zeros(frame2_cropped.shape, frame2_cropped.dtype), 0, -20)
    img_result = final_frame.copy()
    img_result = cv2.cvtColor(final_frame, cv2.COLOR_BGR2BGRA)
    px = coordinates_smooth_final_x[f]
    py = coordinates_smooth_final_y[f]
    # print("frame " + str(f))
    # print(coordinates[f][1])
    print(px, py)
    mouth_area = img_result[py - 60:py + 80, px - 70:px + 70]
    # cv2.circle(img_result, center=tuple([px, py]), radius=5, color=(0, 0, 255), thickness=-1)
    f += 1
    # print(mouth_area.shape)
    # cv2.rectangle(img_result, (px-100, py+100), (px+100, py-100), (255, 0, 0), 2)
    dog_face_no_mouth = cv2.bitwise_and(mouth_area, mouth_area, mask=mouth_mask)
    final_mouth = cv2.add(dog_face_no_mouth, frame2_darker)
    img_result[py - 60:py + 80, px - 70:px + 70] = final_mouth
    cv2.imshow('result', img_result)
    if cv2.waitKey(fps) & 0xFF == ord('q'):
        break
    img_result = cv2.cvtColor(img_result, cv2.COLOR_BGRA2BGR)
    out.write(img_result)
out.release()
cap.release()
cv2.destroyAllWindows()


