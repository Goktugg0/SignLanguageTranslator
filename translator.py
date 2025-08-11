import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    flipped = cv.flip(frame,1)
    
    cv.imshow('frame', flipped)

    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()