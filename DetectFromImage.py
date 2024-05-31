import cv2
import cvzone
from yolov9 import detect_dual


detect_dual.run(weights='D:\yolov8-face-detection\models\yolov9-c-face.pt', source='D:\yolov8-face-detection\images\worlds-largest-selfie.jpg')

