import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from fastai.vision.all import load_learner
from torch.nn import CrossEntropyLoss
from rnn import FrameLSTM

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=2)  # Set maxHands to 2 for detecting both hands

# Load the Fastai model
model_path = "/Users/reetvikchatterjee/Downloads/exported_model_6.pkl"
try:
    learner = load_learner(model_path)
except FileNotFoundError:
    print(f"Error: File '{model_path}' not found.")
    exit()
except Exception as e:
    print(f"Error: Failed to load the model - {str(e)}")
    exit()

# Get the labels used by the model
model_labels = ['Hello', 'How', 'ILoveYou', 'ThankYou', 'You']
# Constants for image processing
offset = 20
imgSize = 300
confidence_threshold = 0.85  # Adjust the threshold as needed

# Initialize variables for storing labelrec values
previous_labels = set()
labelrec_array = ''

while True:
    # Read frame from video capture
    success, img = cap.read()
    if not success:
        print("Failed to read frame")
        break

    # Make a copy of the original image
    imgOutput = img.copy()

    # Find hands in the image
    hands, img = detector.findHands(img)
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']

            # Create a white background image
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Crop and resize hand region
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            if imgCrop.size == 0:
                print("Empty image detected. Skipping...")
                continue

            # Resize hand image to match model input size
            imgCrop = cv2.resize(imgCrop, (imgSize, imgSize))

            # Perform gesture classification
            prediction, _, probabilities = learner.predict(imgCrop)

            # Convert probabilities to a NumPy array for comparison
            probabilities = probabilities.numpy()

            # Check if any label has a probability above the confidence threshold
            if len(probabilities) > 0 and max(probabilities) >= confidence_threshold:
                labels_above_threshold = [model_labels[i] for i, prob in enumerate(probabilities) if prob >= confidence_threshold]

                if labels_above_threshold:
                    # Draw bounding box and label on the output image for each label above threshold
                    for label in labels_above_threshold:
                        if label not in previous_labels:
                            # Append label to the array
                            labelrec_array += label + ' '
                            previous_labels.add(label)

                            # Speak the collected words
                            # ts.text_to_speech(label, language='en', filename='welcome.mp3', slow=False)

                        # Draw bounding box and label on the output image
                        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
                        labelrec = f"{label}"
                        cv2.putText(imgOutput, labelrec, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)
                        print(labelrec)

    # Display output image with bounding box and label
    cv2.imshow('Image', imgOutput)

    # Check for key press and exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
