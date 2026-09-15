import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math 
import time
import pygame
# Step 1: Create a virtual environment, paste in terminal: python -m venv handenv
# Step 2: Install dependencies, paste in terminal: pip install -r requirements.txt

# To run environment, paste: handenv\Scripts\activate
# To run code, paste: python test.py

# Capture video from the default camera
cap = cv2.VideoCapture(0)

detector = HandDetector(
    maxHands=2,
    detectionCon=0.3,
    minTrackCon=0.3,
    )  
#Array of hand landmarks, used to draw lines between them to create a hand skeleton
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (0,9),(9,10),(10,11),(11,12),   # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20), # pinky
    (5,9),(9,13),(13,17)            # palm
]
prev_left = None
prev_right = None

counter = 0
offset = 20
imgSize = 300
# Load the classifiers and labels
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = ["Infinite Void", "Nothing"]
classifier2 = Classifier("Model2/keras_model.h5", "Model2/labels.txt")
labels2 = ["Malevolent Shrine", "Idle Death Gamble"]

current_label = None
start_time = None
confirmed = False
# Initialize Pygame mixer for audio playback and load audio files
pygame.mixer.init()
audio_playing = False
gojoaudio = pygame.mixer.Sound("Audio/Gojo Domain.mp3")
sukunaaudio = pygame.mixer.Sound("Audio/Sukuna Domain.mp3")
mahoragaaudio = pygame.mixer.Sound("Audio/Mahoraga Audio.mp3")
# Function to apply a tint to the image, currently not used for all but can be used to create a visual effect when a domain is confirmed
def applyTint(img, color, alpha):
    overlay = np.ones_like(img) * np.array(color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

while True:
    # Read a frame from the video capture
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, draw=False, flipType=True)

    leftHand = None
    rightHand = None

    if hands:   
        for hand in hands:
            lmList = hand["lmList"]
            for start, end in HAND_CONNECTIONS:
                x1, y1, _ = lmList[start]
                x2, y2, _ = lmList[end]
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            for x, y, z in hand["lmList"]:
                cv2.circle(img, (x, y), 5, (0, 0, 255), cv2.FILLED)
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
      #Image Preprocessing for single hand, used to prepare the image for training and prediction, includes cropping, resizing, and centering the hand in a white background
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
        #
        prediction, index = classifier.getPrediction(imgWhite)
        predicted_label = labels[index]
        #Checks if the predicted label is "Infinite Void" and if confirmed, applies tint and plays audio
        if predicted_label == current_label == "Infinite Void":
            cv2.putText(img, predicted_label, (20, 50),
                          cv2.FONT_HERSHEY_TRIPLEX, 1.5, (191, 0, 95), 2, cv2.LINE_AA)
            if confirmed:
                if confirmationTime is not None and time.time() - confirmationTime >= 7:
                    applyTint(img, (0, 0, 0), 0.5)
            if start_time is not None and not confirmed:
                elapsed = time.time() - start_time
                if elapsed >= 2:  # Confirm after 2 seconds
                    confirmed = True
                    gojoaudio.play()
                    confirmationTime = time.time()
                    audio_playing = True 
                    
        #Otherwise, reset current label and stops audio
        else:               
                current_label = predicted_label
                start_time = time.time()
                confirmed = False
                audio_playing = False
                gojoaudio.stop()
                sukunaaudio.stop()
                mahoragaaudio.stop()
        #If audio playing and audio has finished, reset variables
        if audio_playing == True:
            if not pygame.mixer.get_busy():
                audio_playing = False
                confirmed = False
                start_time = None
                current_label = None

    #    if imgCrop.size != 0:
     #           cv2.imshow("SingleHandCrop", imgCrop)
      #          cv2.imshow("WhiteBackground", imgWhite)

          
    #Image preprocessing for two hands, similar to single hand but crops both hands together and prepares the image for training and prediction
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
            imgWhite[:, wGap:wCal+wGap] = imgResize

        else:
            k = imgSize / w
            hCal =  math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap+hCal, :] = imgResize

        prediction, index = classifier2.getPrediction(imgWhite)
        predicted_label = labels2[index]
        #Checks if the predicted label is "Malevolent Shrine" and if confirmed, applies tint and plays audio
        if predicted_label == current_label == "Malevolent Shrine":
            cv2.putText(img, predicted_label, (20, 50),
                          cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
            if confirmed:
                if confirmationTime is not None and time.time() - confirmationTime >= 6:
                    applyTint(img, (0, 0, 138), 0.5)
            if start_time is not None and not confirmed:
                elapsed = time.time() - start_time
                if elapsed >= 2:  # Confirm after 2 seconds
                    confirmed = True
                    print(f"Confirmed: {predicted_label}")
                    sukunaaudio.play()
                    confirmationTime = time.time()
                    audio_playing = True 
#This one was originally supposed to be idle death gamble, but there was no audio for it since it wasnt in the anime yet
#The prediction works with the mahoraga hand sign, so i just changed everything to that
        elif predicted_label == current_label == "Idle Death Gamble":
            #This makes text with an outline by overlaying 2 texts
            cv2.putText(img, "Mahoraga", (20, 50),
                          cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, "Mahoraga", (20, 50),
                          cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            if confirmed:
                if confirmationTime is not None and time.time() - confirmationTime >= 7:
                    applyTint(img, (0, 0, 0), 0.5) #Tint is "color, alpha/opacity"
            if start_time is not None and not confirmed:
                  elapsed = time.time() - start_time
                  if elapsed >= 2:  # Confirm after 2 seconds
                        confirmed = True
                        print(f"Confirmed: {predicted_label}")
                        mahoragaaudio.play()
                        confirmationTime = time.time()
                        audio_playing = True
#same function as the one for single hand
        else:               
                current_label = predicted_label
                start_time = time.time()
                confirmed = False
                audio_playing = False
                gojoaudio.stop()
                sukunaaudio.stop()
                mahoragaaudio.stop()

        if audio_playing == True:
            if not pygame.mixer.get_busy():
                audio_playing = False
                confirmed = False
                start_time = None
                current_label = None

#this just shows the other two windows for hand crops, used for creating image data for training
      #  if imgCrop.size != 0:
      #      cv2.imshow("BothHandsCrop", imgCrop) 2 hands window
      #      cv2.imshow("WhiteBackground", imgWhite) img white


    # Display the captured frame
    cv2.imshow('Image', img) #main window
    
    # Wait for a key press and check if it's 'q' to exit
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

