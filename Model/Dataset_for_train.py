import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Constants for image processing
offset = 20
imgSize = 300
counter = 0

# Folder to save images
folder = "/Users/reetvikchatterjee/Desktop/Dataset/ThankYou"

while True:
    # Read frame from video capture
    success, img = cap.read()
    if not success:
        print("Failed to read frame")
        break

    # Find hands in the image
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop and resize hand region
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        if imgCrop.size == 0:
            print("Empty image detected. Skipping...")
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        # Display cropped and resized hand image
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    # Display original image
    cv2.imshow('Image', img)

    # Check for key press
    key = cv2.waitKey(1)
    if key == ord("s"):
        # Save the image
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    elif key == ord("q"):  # Press 'q' to quit
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
