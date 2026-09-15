import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math 
import time

# Capture video from the default camera
cap = cv2.VideoCapture(0)

detector = HandDetector(
    maxHands=2,
    detectionCon=0.3,
    minTrackCon=0.3,
    ) 
prev_left = None
prev_right = None

folder = "Data/Nothing"
counter = 0

offset = 20
imgSize = 300

while True:
    # Read a frame from the video capture
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    leftHand = None
    rightHand = None

    if hands:   
        for hand in hands:
            if hand['type'] == 'Left':
                leftHand = hand
            elif hand['type'] == 'Right':
                rightHand = hand

    if leftHand is None:
        leftHand = prev_left
    if rightHand is None:
        rightHand = prev_right

    prev_left = leftHand
    prev_right = rightHand

    if len(hands) == 1:
        x, y, w, h = hands[0]['bbox']
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        imgWhite = np.ones((imgSize, imgSize, 3 ), np.uint8) * 255

        if leftHand:
            lmList = leftHand["lmList"]
            for x, y, z in lmList:
                cv2.circle(imgWhite, (x, y), 5, (0,0,0), cv2.FILLED)

        if rightHand:
            lmList = rightHand["lmList"]
            for x, y, z in lmList:
                cv2.circle(imgWhite, (x, y), 5, (0,0,0), cv2.FILLED)


        imgCropshape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize

        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hGap + hCal, :] = imgResize
        
        
        if imgCrop.size != 0:
                cv2.imshow("SingleHandCrop", imgCrop)
                cv2.imshow("WhiteBackground", imgWhite)

          

    if len(hands) == 2:
        x1, y1, w1, h1 = hands[0]['bbox']
        x2, y2, w2, h2 = hands[1]['bbox']

        x_min = min(x1, x2) - offset
        y_min = min(y1, y2) - offset
        x_max = max(x1+w1, x2+w2) + offset
        y_max = max(y1+h1, y2+h2) + offset

        h_img, w_img, _ = img.shape
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w_img, x_max)
        y_max = min(h_img, y_max)

        imgWhite = np.ones((imgSize, imgSize, 3 ), np.uint8) * 255

        imgCrop = img[y_min:y_max, x_min:x_max]

        imgCropshape = imgCrop.shape

        h, w = imgCrop.shape[:2]

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap+wCal] = imgResize
            imgWhite[:, wGap:wCal+wGap] = imgResize

        else:
            k = imgSize / w
            hCal =  math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap+hCal, :] = imgResize

        if imgCrop.size != 0:
            cv2.imshow("BothHandsCrop", imgCrop)
            cv2.imshow("WhiteBackground", imgWhite)


    # Display the captured frame
    cv2.imshow('Image', img)
    
    # Wait for a key press and check if it's 'q' to exit or 's'/'a' to save the image
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('s') or key == ord('a'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
        if counter == 300:
            break

