import mediapipe as mp
import cv2 as cv


class HandDetector:
    def __init__(self, static_image_mode=False,
                      max_num_hands=2,
                      model_complexity=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.cap = cv.VideoCapture(0)

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode,self.max_num_hands,
                                        self.model_complexity, self.min_detection_confidence,
                                        self.min_tracking_confidence)

        self.mpDraw = mp.solutions.drawing_utils  # For drawing landmarks

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        frame = cv.flip(frame, 1)
        # Convert BGR (OpenCV format) to RGB (MediaPipe format)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Process with the hands
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, lm, self.mpHands.HAND_CONNECTIONS)
        return frame, results

    def release(self):
        self.cap.release()
        cv.destroyAllWindows()


def main():
    detector = HandDetector()
    while True:
        frame, results = detector.read_frame()

        if frame is None:
            break

        # Example: print number of detected hands
        if results.multi_hand_landmarks:
            print(results.multi_hand_landmarks)

        cv.imshow('frame', frame)

        if cv.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    main()