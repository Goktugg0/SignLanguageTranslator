import pickle
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import mediapipe as mp

from handDetector import HandDetector

"Will load the model and test it in real time"

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

data = pd.read_csv("sign_data.csv")

# Get unique labels in sorted order
class_names = sorted(data['label'].unique().tolist())

detector = HandDetector()

while True:
    frame, results = detector.read_frame()

    if frame is None:
        break

    landmark_vector = detector.landmarks_to_list(results)

    if landmark_vector is not None:
        data = np.array(landmark_vector[:63]).reshape(1, 63)
        prediction = model.predict(data)

        pred_index = np.argmax(prediction)  # index of highest probability
        confidence = np.max(prediction)  # probability value
        pred_label = labels[pred_index]  # map index → label

        print(f"Prediction: {pred_label} (confidence: {confidence:.2f})")
    cv.imshow("Sign Language Translator", frame)

    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):  # Quit
        break

detector.release()
