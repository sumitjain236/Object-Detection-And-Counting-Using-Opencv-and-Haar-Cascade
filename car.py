import numpy as np
import cv2

car_cascade = cv2.CascadeClassifier('mycascade.xml')

cap = cv2.VideoCapture("cars2.mp4")

while cap.isOpened():
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.3, 2)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w+5,y+h+5),(0,0,255),3)

    cv2.imshow('Car Detector',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    



cap.release()

cv2.destroyAllWindows()
