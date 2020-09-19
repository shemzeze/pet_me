import cv2
import numpy as np
import dlib


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while (True):
    ret, frame = cap.read()
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
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_points.append((x, y))

            # cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        points = np.array(landmark_points, np.int32)
        convexhull = cv2.convexHull(points)

        cv2.polylines(frame, [convexhull], True, (0, 255, 0), 3)
        # cv2.fillConvexPoly(mask, convexhull, 255)

    # mouth_mask = cv2.bitwise_and(frame, frame, mask=mask)


    cv2.imshow('frame', frame)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("mouth mask", mouth_mask)
    if (cv2.waitKey(20) & 0xFF == ord('q')):
        break


cap.release()
cv2.destroyAllWindows()
