import socket
import dlib, cv2, sys
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from math import hypot, atan2, degrees
from vidstab import VidStab
from moviepy.editor import *
from moviepy import video


def angle_between(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return degrees(atan2(yDiff, xDiff))


def find_dog_face(vid):
    print("finding pet Landmarks")
    clip = VideoFileClip("VID_PET.mp4")
    clip = video.fx.all.resize(clip, (320, 640))
    clip.write_videofile("VID_PET_SMALL.mp4")
    SCALE_FACTOR = 0.2
    vid = sys.argv[0]
    cap = cv2.VideoCapture("VID_PET_SMALL.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')
    predictor = dlib.shape_predictor('landmarkDetector.dat')
    nose_coordinates = np.empty((0, 2), dtype=int)
    left_eye_coordinates = np.empty((0, 2), dtype=int)
    right_eye_coordinates = np.empty((0, 2), dtype=int)
    eyes_width_arr = np.empty(1, dtype="int32")
    angle_eyes = []


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
        print("detector working")

        for i, d in enumerate(roi):
            x1, y1 = int(d.rect.left() / SCALE_FACTOR), int(d.rect.top() / SCALE_FACTOR)
            x2, y2 = int(d.rect.right() / SCALE_FACTOR), int(d.rect.bottom() / SCALE_FACTOR)
            shape = predictor(frame_gray, d.rect)
            shape = face_utils.shape_to_np(shape)
            nose = shape[3]
            print(nose)
            nose_coordinates = np.append(nose_coordinates, [nose], axis=0)
            left_eye = shape[5]
            left_eye_coordinates = np.append(left_eye_coordinates, [left_eye], axis=0)
            right_eye = shape[2]
            right_eye_coordinates = np.append(right_eye_coordinates, [right_eye], axis=0)
            eyes_width = int(hypot(int(left_eye[1] - right_eye[1]), int(left_eye[0] - right_eye[0])))
            eyes_width_arr = np.append(eyes_width_arr, [eyes_width], axis=0)
            angle = angle_between(left_eye, right_eye)
            angle_eyes.append(angle)
            cv2.circle(img_result, center=tuple((nose / SCALE_FACTOR).astype(int)), radius=5, color=(255, 0, 0),
                       thickness=-1, lineType=cv2.LINE_AA)
        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGRA2BGR)
        cv2.imshow('result', img_result)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    eyes_width_arr = eyes_width_arr[1:]
    # print(angle_eyes)
    # print(eyes_width_arr)
    return nose_coordinates * 5, eyes_width_arr * 5, angle_eyes


s = socket.socket()
host = socket.gethostname()
ip_address = "192.168.0.12"
BUFFER_SIZE = 1024
port = 8080
s.bind((host, port))

while True:
    print("waiting for someone to connect...")
    s.listen(1)
    conn, addr = s.accept()
    print(addr, "has connected to me")
    client_ip, client_port = addr
    with open('VID_PET.mp4', 'wb') as f:
        while True:
            print('receiving data...')
            data = conn.recv(BUFFER_SIZE)
            print('data=%s', data)
            if not data:
                f.close()
                print('file close()')
                break
            f.write(data)
    f.close()
    print('Successfully get the file')
    coordinates, eyes_hypot, angle_mouth = find_dog_face("VID_PET.mp4")
    coordinates_x = coordinates[:, 0]
    coordinates_y = coordinates[:, 1]
    x_smooth_float = savgol_filter(coordinates_x, 31, 3, mode="nearest")
    coordinates_smooth_x = np.empty(1, dtype="int32")
    coordinates_smooth_y = np.empty(1, dtype="int32")
    eyes_hypot_smooth = np.empty(1, dtype="int32")
    for x in x_smooth_float:
        x = int(x)
        coordinates_smooth_x = np.append(coordinates_smooth_x, [x], axis=0)
    y_smooth_float = savgol_filter(coordinates_y, 31, 3, mode="nearest")
    for y in y_smooth_float:
        y = int(y)
        coordinates_smooth_y = np.append(coordinates_smooth_y, [y], axis=0)

    eyes_hypot_smooth_float = savgol_filter(eyes_hypot, 31, 3, mode="nearest")
    for w in eyes_hypot_smooth_float:
        w = int(w)
        eyes_hypot_smooth = np.append(eyes_hypot_smooth, [w], axis=0)
    angle_mouth_smooth_float = savgol_filter(angle_mouth, 31, 3, mode="nearest")

    # plt.plot(angle_mouth_smooth_float, color="red")
    # # # plt.plot(coordinates_y, color="red")
    # plt.plot(angle_mouth, color="green")
    # # plt.plot(coordinates_smooth_y[1:], color="green")
    # print(eyes_hypot_smooth[1:])
    # # print(coordinates_smooth_y)
    # plt.show()
    coordinates_smooth_final_x = coordinates_smooth_x[1:]
    coordinates_smooth_final_y = coordinates_smooth_y[1:]
    eyes_hypot_smooth = eyes_hypot_smooth[1:]
    s.close()
    print('connection closed')
    break


s2 = socket.socket()
host = socket.gethostname()
ip_address = "192.168.0.12"
BUFFER_SIZE = 1024
port = 8080
s2.bind((host, port))

while True:
    s2.listen(1)
    print("waiting for someone to connect...")
    conn, addr = s2.accept()
    print(addr, "has connected to me")
    client_ip, client_port = addr
    with open('VID_FACE.mp4', 'wb') as f:
        while True:
            print('receiving data...')
            data = conn.recv(BUFFER_SIZE)
            print('data=%s', data)
            if not data:
                f.close()
                print('file close()')
                break
            # write data to a file
            f.write(data)
    f.close()
    print('Successfully get the file')
    clip2 = VideoFileClip("VID_FACE.mp4")
    clip2 = video.fx.all.resize(clip2, (320, 640))
    clip2.write_videofile("VID_FACE_SMALL.mp4")
    print("Stabilizing face vid")
    stabilizer = VidStab()
    stabilizer.stabilize(input_path="VID_FACE_SMALL.mp4", output_path="VID_FACE_STAB.mp4", output_fourcc="mp4v",
                         smoothing_window=100)
    print("Video is stabilized")
    cap2 = cv2.VideoCapture("VID_FACE_STAB.mp4")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sourcePath = "Stabilize head.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("VID_MOUTH_CROP.mp4", fourcc, 30, (width, height))
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        faces = detector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,10), 5)
            landmarks = predictor(gray, face)
            landmark_points = []
            for n in range(48, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmark_points.append((x, y))
            # cv2.rectangle(frame_rot, (top_left_x, top_left_y), (bot_left_x, bot_left_y), (255,0,10), 5)
            points = np.array(landmark_points, np.int32)
            convexhull = cv2.convexHull(points)
            # cv2.polylines(frame, [convexhull], True, (0, 255, 0), 3)
            mouth_fill = cv2.fillConvexPoly(mask, convexhull, 255)
            kernel = np.ones((10, 10), np.float32) / 100
            dst = cv2.filter2D(mouth_fill, -1, kernel)
            mask = cv2.dilate(mouth_fill, kernel, dst)
            mask = cv2.blur(mouth_fill, (10, 10))

        mouth_mask = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("mouth mask", mouth_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(mouth_mask)
    cap2.release()
    out.release()
    cv2.destroyAllWindows()
    s2.close()
    print('connection closed')
    break


cap3 = cv2.VideoCapture("VID_PET_SMALL.mp4")
cap4 = cv2.VideoCapture("VID_MOUTH_CROP.mp4")
f = 0
fps = int(cap3.get(cv2.CAP_PROP_FPS))
width3 = int(cap3.get(cv2.CAP_PROP_FRAME_WIDTH))
height3 = int(cap3.get(cv2.CAP_PROP_FRAME_HEIGHT))
width4 = int(cap4.get(cv2.CAP_PROP_FRAME_WIDTH))
height4 = int(cap4.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("FINAL_NO_SOUND.mp4", fourcc, 30, (width, height))

while True:
    ret, final_frame = cap3.read()
    ret, frame2 = cap4.read()
    if not ret:
        break
    px = coordinates_smooth_final_x[f]
    py = coordinates_smooth_final_y[f]
    eyes_hypot_smooth_width = eyes_hypot_smooth[f]
    eyes_hypot_smooth_height = eyes_hypot_smooth_width
    frame2_alpha = cv2.cvtColor(frame2, cv2.COLOR_BGR2BGRA)
    # print(frame2.shape)
    # frame2_resize = cv2.resize(frame2_alpha, (width4 // 3, height4 // 3))
    frame2_cropped = frame2_alpha[300:440, 80:220]
    frame2_resize_from_eyes = cv2.resize(frame2_cropped, (eyes_hypot_smooth_width, eyes_hypot_smooth_height))
    # M = cv2.getRotationMatrix2D((px, py), angle_mouth_smooth_float[f], 1)
    # frame2_rot = cv2.warpAffine(frame2_resize_from_eyes, M, (frame2_resize_from_eyes.shape[1], frame2_resize_from_eyes.shape[0]))
    frame2_gray = cv2.cvtColor(frame2_resize_from_eyes, cv2.COLOR_BGRA2GRAY)
    _, mouth_mask = cv2.threshold(frame2_gray, 2, 255, cv2.THRESH_BINARY_INV)
    frame2_darker = cv2.addWeighted(frame2_resize_from_eyes, 1, np.zeros(frame2_resize_from_eyes.shape, frame2_resize_from_eyes.dtype), 0, -10)
    img_result = final_frame.copy()
    img_result = cv2.cvtColor(final_frame, cv2.COLOR_BGR2BGRA)

    # to change the location of the mouth, change the (int(eyes_hypot_smooth_width // x - the higher x the lower the mouth
    # top_left_mouth_area = (px-(eyes_hypot_smooth_width // 2), py-(int(eyes_hypot_smooth_width // 5)))
    # bottom_right_mouth_area = (px+(eyes_hypot_smooth_width // 2), py+(eyes_hypot_smooth_width // 2))
    top_left_mouth_area = (px-(eyes_hypot_smooth_width // 2), py)
    bottom_right_mouth_area = (px+(eyes_hypot_smooth_width // 2), py+eyes_hypot_smooth_width)
    mouth_area = img_result[top_left_mouth_area[1]:top_left_mouth_area[1] + eyes_hypot_smooth_height, top_left_mouth_area[0]:top_left_mouth_area[0] + eyes_hypot_smooth_width]
    # cv2.circle(img_result, center=tuple([px, py]), radius=5, color=(0, 0, 255), thickness=-1)
    # # cv2.rectangle(img_result, (px-100, py+100), (px+100, py-100), (255, 0, 0), 2)
    dog_face_no_mouth = cv2.bitwise_and(mouth_area, mouth_area, mask=mouth_mask)
    final_mouth = cv2.addWeighted(dog_face_no_mouth, 1, frame2_resize_from_eyes, 1, 1)
    img_result[top_left_mouth_area[1]:top_left_mouth_area[1] + eyes_hypot_smooth_height, top_left_mouth_area[0]:top_left_mouth_area[0] + eyes_hypot_smooth_width] = final_mouth
    cv2.imshow('result', img_result)
    f += 1
    # cv2.imshow('result2', dog_face_no_mouth)
    if cv2.waitKey(fps) & 0xFF == ord('q'):
        break
    img_result = cv2.cvtColor(img_result, cv2.COLOR_BGRA2BGR)
    out.write(img_result)
out.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()


clip = VideoFileClip("FINAL_NO_SOUND.mp4")
clip2 = VideoFileClip("VID_FACE.mp4")
clip3 = VideoFileClip("VID_PET.mp4")
# clip2 = clip2.set_fps(30.0)
aclip = clip.audio
aclip2 = clip2.audio
aclip3 = clip3.audio
sound_both2 = CompositeAudioClip([aclip3, aclip2])
clip.audio = sound_both2
clip.write_videofile("FINAL_PETME.mp4")

s3 = socket.socket()
host = socket.gethostname()
ip_address = "192.168.0.12"
BUFFER_SIZE = 1024
port = 8080
s3.bind((host, port))

while True:
    s3.listen(1)
    print("waiting for someone to connect...")
    conn, addr = s3.accept()
    print('Got connection from', addr)
    # data = conn.recv(1024)
    # print('Server received', repr(data))

    # f = open(filename, 'rb')
    with open('FINAL_PETME.mp4', 'rb') as f:
        lll = f.read(1024)
        while lll:
            data = conn.send(lll)
            print('Sent ', repr(lll))
            print(len(lll))
            lll = f.read(1024)
            # if not data:
            #     break
    f.close()
    print('Done sending')
    conn.close()
    s2.close()
    print('connection closed')
    break

