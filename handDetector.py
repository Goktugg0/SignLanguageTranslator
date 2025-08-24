import csv
import cv2 as cv
import mediapipe as mp
import os

class HandDetector:
    """"
    The class to create hand landmarks and collects it with openCv and mediapipe.
    It can return the raw frame with landmarks drawn, and convert detected landmarks
    into a flattened list. Also, the class itself allows for further project creation involving hand
    and/or arm tracker, such as trained sport coaches for individuals.
    """
    def __init__(self, static_image_mode=False,
                      max_num_hands=1,
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
        self.sequence = []  # Stores the landmark frames for one gesture
        self.data_dir = "collected_signs"
        os.makedirs(self.data_dir, exist_ok=True)

    def read_frame(self):
        """This function is to convert the data to mediapipe format and check for sensible values.
           Returns the converted result and the Handmarks drawn frame"""

        ret, frame = self.cap.read()
        if not ret:
            return None, None

        # Convert BGR (OpenCV format) to RGB (MediaPipe format)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Process with the hands
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, lm, self.mpHands.HAND_CONNECTIONS)
        return frame, results

    @staticmethod
    def landmarks_to_list(results):
        """Return flattened list of 126 values (2 hands * 21 landmarks * 3 cords).
           If a hand is missing, fill with zeros. It is called with objects but can be independent
           of any objects."""
        if results.multi_hand_landmarks:
            hands_data = []
            for handLms in results.multi_hand_landmarks:
                for lm in handLms.landmark:
                    hands_data.extend([lm.x, lm.y, lm.z])
            # If only one hand, pad with zeros
            if len(results.multi_hand_landmarks) == 1:
                hands_data.extend([0.0] * (21 * 3))
            output = hands_data
        else:
            # No hands detected → all zeros
            output = [0.0] * (21 * 3 * 2)
        return output


    def release(self):
        # Releases the use of the cap and destroys on exit
        self.cap.release()
        cv.destroyAllWindows()


def main():
    SIGN_LABEL = "u"  # For specifying each sign
    file_path = "sign_data.csv" # Collected csv data for training

    detector = HandDetector()

    # Create CSV header if file doesn't exist
    if not os.path.exists(file_path):
        header = ["label"] + [f"{axis}{i}" for i in range(1, 64) for axis in ("x", "y", "z")]
        with open(file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    while True:
        frame, results = detector.read_frame()

        cv.putText(frame, f"Collecting: {SIGN_LABEL}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Checking in each tick
        if frame is None:
            break

        if results.multi_hand_landmarks:
            print(results.multi_hand_landmarks)

        cv.imshow('frame', frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord('s'):  # Save current frame landmarks
            row = [SIGN_LABEL] + detector.landmarks_to_list(results)
            with open(file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"Saved one sample for {SIGN_LABEL}")

        if key == ord('q'):  # Quit
            break

    detector.release()

if __name__ == '__main__':
    main()