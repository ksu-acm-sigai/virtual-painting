import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm


folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# Setting up hand tracking module
detector = htm.handDetector(detectionCon=0.85)

while True:
    # 1) Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    # 2) Find Hand Landmarks
    img = detector.findHands(img)
    
    # 3) Check which fingers are up
    # 2 fingers up = don't draw
    # 1 finger up = draw
    
    
    img[0:125, 0:1280] = header
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)