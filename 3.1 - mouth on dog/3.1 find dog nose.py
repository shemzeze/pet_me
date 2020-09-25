import dlib, cv2, sys
from imutils import face_utils
import numpy as np
vid = "Dog small.mp4"


SCALE_FACTOR = 0.2
dog_path = vid
dog_path = sys.argv[0]
cap = cv2.VideoCapture("Dog small.mp4")
cap2 = cv2.VideoCapture("Mouth_crop.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('PetMe.mp4', fourcc, 30, (width, height))
detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')
predictor = dlib.shape_predictor('landmarkDetector.dat')

outer_loop = 0
inner_loop = 0
inner_sub_loop = 0
img_result2 = None
while True:
    outer_loop += 1
    # print("rendering frame number: " + str(outer_loop))

    ret, frame = cap.read()
    ret, frame2 = cap2.read()
    frame2_alpha = cv2.cvtColor(frame2, cv2.COLOR_BGR2BGRA)
    frame2_resize = cv2.resize(frame2_alpha, (width2//3, height2//3))
    print(frame2_resize.shape)
    frame2_cropped = frame2_resize[140:280, 250:390]
    frame2_gray = cv2.cvtColor(frame2_cropped, cv2.COLOR_BGR2GRAY)
    _, mouth_mask = cv2.threshold(frame2_gray, 5, 255, cv2.THRESH_BINARY_INV)
    img_result = frame.copy()
    img_result = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, dsize=None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(frame_gray, upsample_num_times=1)

    for i, d in enumerate(dets):
        inner_loop += 1
        x1, y1 = int(d.rect.left() / SCALE_FACTOR), int(d.rect.top() / SCALE_FACTOR)
        x2, y2 = int(d.rect.right() / SCALE_FACTOR), int(d.rect.bottom() / SCALE_FACTOR)
        # cv2.rectangle(img_result, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
        # print(x1, y1, x2, y2)
        shape = predictor(frame_gray, d.rect)
        shape = face_utils.shape_to_np(shape)

        for m, p in enumerate(shape):
            inner_sub_loop += 1
            p = shape[3]
            py, px = (p / SCALE_FACTOR).astype(int)
            print(px, py)
            mouth_area = img_result[px-60:px+80, py-70:py+70]
            # print(mouth_area.shape, mouth_mask.shape)
            # cv2.rectangle(img_result, (px-100, py+100), (px+100, py-100), (255, 0, 0), 2)
            # # cv2.putText(img_result, "nose", tuple((p / SCALE_FACTOR).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            dog_face_no_mouth = cv2.bitwise_and(mouth_area, mouth_area, mask=mouth_mask)
            # # dog_face_no_mouth = cv2.cvtColor(dog_face_no_mouth, cv2.COLOR_BGRA2BGR)
            final_mouth = cv2.add(dog_face_no_mouth, frame2_cropped)
            img_result[px - 60:px + 80, py - 70:py + 70] = final_mouth
    img_result = cv2.cvtColor(img_result, cv2.COLOR_BGRA2BGR)
    cv2.imshow('result', img_result)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

    if img_result2 is not None:
        out.write(img_result2)


cap.release()
out.release()
cv2.destroyAllWindows()

