import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()