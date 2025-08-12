import mediapipe as mp
import cv2 as cv

cap = cv.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      model_complexity=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5
                      )


while True:
    ret, frame = cap.read()

    mirrored_frame = cv.flip(frame, 1)

    # Convert BGR (OpenCV format) to RGB (MediaPipe format)
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # Process with the hands
    results = hands.process(frameRGB)

    print(results.multi_hand_landmarks)

    cv.imshow('frame', mirrored_frame)

    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()