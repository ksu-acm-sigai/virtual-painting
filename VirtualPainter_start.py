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

    img[0:125, 0:1280] = header
    
    # 3) Check which fingers are up
    # 2 fingers up = don't draw
    # 1 finger up = draw

    lmList, bbox = detector.findPosition(img, draw=False)
    # if (lmList):
    #     print(lmList[4])
    #     print(lmList[8])
    #     print(lmList[12])
    #     print(lmList[16])
    #     print(lmList[20])
    # print("=========================")

    handLandmarks = detector.findHandLandMarks(image=img, draw=True)

    if lmList:
        # Find how many fingers are up
        fingers = detector.fingersUp()
        totalFingers = fingers.count(1)
        print(fingers)
        '''
        cv2.putText(img, f'Fingers:{totalFingers}', (bbox[0] + 200, bbox[1] - 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        '''
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)