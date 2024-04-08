import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=2)

# Constants for image processing
offset = 20
imgSize = 300
counter = 0

# Folder to save images
folder = "/Users/reetvikchatterjee/Desktop/Dataset/How"

while True:
    # Read frame from video capture
    success, img = cap.read()
    if not success:
        print("Failed to read frame")
        break

    # Find hands in the image
    hands, img = detector.findHands(img)

    if len(hands) == 2:
        # Get bounding box for both hands
        bbox1 = hands[0]['bbox']
        bbox2 = hands[1]['bbox']

        # Calculate bounding box that covers both hands
        x_min = min(bbox1[0], bbox2[0])
        y_min = min(bbox1[1], bbox2[1])
        x_max = max(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y_max = max(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

        # Expand the bounding box to ensure both hands are included with an offset
        x_min = max(0, x_min - offset)
        y_min = max(0, y_min - offset)
        x_max = min(img.shape[1], x_max + offset)
        y_max = min(img.shape[0], y_max + offset)

        # Create a white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop and resize hand region
        imgCrop = img[y_min:y_max, x_min:x_max]
        if imgCrop.size == 0:
            print("Empty image detected. Skipping...")
            continue

        # Resize hand image to match model input size
        imgResize = cv2.resize(imgCrop, (imgSize, imgSize))

        # Display cropped and resized hand image
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgResize)

        # Check for key press
        key = cv2.waitKey(1)
        if key == ord("s"):
            # Save the image
            counter += 1
            cv2.imwrite(f'{folder}/Image_{counter}_{time.time()}.jpg', imgResize)
            print(f"Image saved: Image_{counter}_{time.time()}.jpg")

    # Display original image
    cv2.imshow('Image', img)

    # Check for key press
    if cv2.waitKey(1) == ord("q"):  # Press 'q' to quit
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
