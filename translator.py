import pickle

import cv2 as cv
import keras
import numpy as np

from handDetector import HandDetector

model = keras.models.load_model("model.keras")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

detector = HandDetector()

while True:
    frame, results = detector.read_frame()
    if frame is None:
        break

    landmark_vector = detector.landmarks_to_list(results)


    if landmark_vector is not None and len(landmark_vector) >= 63:
        # Convert to array and reshape
        landmarks = np.array(landmark_vector[:63], dtype=np.float32).reshape(1, 63)

        # Scale features using the saved scaler
        landmarks_scaled = scaler.transform(landmarks)

        # Predict
        predicted = model.predict(landmarks_scaled, verbose=0)
        predicted_index = np.argmax(predicted)
        predicted_label = le.inverse_transform([predicted_index])[0]  # decode label
        confidence = np.max(predicted)

        text = f"{predicted_label} ({confidence:.2f})"
        cv.putText(frame, text, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.imshow("Sign Language Translator", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

detector.release()

