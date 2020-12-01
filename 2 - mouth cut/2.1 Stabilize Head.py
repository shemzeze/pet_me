import cv2
import numpy as np
from vidstab import VidStab

stabilizer = VidStab()
stabilizer.stabilize(input_path="Mouth.mp4", output_path="Stablize head.mp4", output_fourcc="mp4v", smoothing_window=100)
print("Video is stabilized")