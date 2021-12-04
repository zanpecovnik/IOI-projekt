import cv2
import time
import os

camWidth, camHeight = 1280, 720
imageDir = "images"

overlayImageSize = 180
overlayImage = cv2.imread(f'{imageDir}/square.jpg')
overlayImagePositions = [
    {'x': 120, 'y': 60},
    {'x': 300, 'y': 440},
    {'x': 520, 'y': 270},
    {'x': 760, 'y': 150},
    {'x': 890, 'y': 540}
]

cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

while True:
    success, img = cap.read()

    for position in overlayImagePositions:
        img[position['y'] : position['y'] + overlayImageSize, position['x'] : position['x'] + overlayImageSize] = overlayImage

    cv2.imshow("Image", img)
    cv2.waitKey(1)