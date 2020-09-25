import cv2
import numpy as np

cap = cv2.VideoCapture("Mouth_crop.mp4")

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter("Mouth color correct.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_darker = cv2.addWeighted(frame, 1, np.zeros(frame.shape, frame.dtype), 0, -20)
    cv2.imshow("frame", frame_darker)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
    out.write(frame_darker)
cap.release()
out.release()
cv2.destroyAllWindows()