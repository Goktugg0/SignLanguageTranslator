import pickle
from pathlib import Path

import cv2 as cv
import keras
import numpy as np
import pandas as pd
import mediapipe as mp

from handDetector import HandDetector

"Will load the model and test it in real time"

# Load the model
model = keras.models.load_model("model.keras")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

data = pd.read_csv("sign_data.csv")

# Get unique labels in sorted order
class_names = sorted(data['label'].unique().tolist())

detector = HandDetector()

while True:
    frame, results = detector.read_frame()

    if frame is None:
        break

    landmark_vector = detector.landmarks_to_list(results)

    cv.imshow("Sign Language Translator", frame)

    if landmark_vector is not None:
        landmarks = np.array(landmark_vector[:63], dtype=np.float32).reshape(1, 63)
        landmarks_scaled = scaler.transform(landmarks)
        predicted = model.predict(landmarks_scaled, verbose=0)
        pred_index = np.argmax(predicted)
        pred_label = class_names[pred_index]
        confidence = np.max(predicted)

        print(f"Prediction: {pred_label} ({confidence:.2f})")

    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):  # Quit
        break

detector.release()
