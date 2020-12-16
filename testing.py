import numpy as np
import cv2
from imutils.video import FPS
from numba.cuda.simulator import kernel



tracker = cv2.TrackerCSRT_create()
car_cascade = cv2.CascadeClassifier('mycascade.xml')

min_contour_width=20
min_contour_height=20
matches =[]
cap = cv2.VideoCapture("cars2.mp4")
count=0
offset=50      #10
line_height=450
def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1
    return cx,cy

cap.set(3,1920)
cap.set(4,1080)
while cap.isOpened():
    ret, img = cap.read()
    #print(img)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.3, 3)
    cv2.line(img, (0, line_height), (1200,line_height), (0, 255, 0), 3)
    #blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    #dilated = cv2.dilate(th, np.ones((3, 3)))
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    #closing = cv2.morphologyEx(dilated,cv2.MORPH_CLOSE,kernel)
    #contours, h = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #for (i, c) in enumerate(contours):
        #(x, y, w, h) = cv2.boundingRect(c)
        #contour_valid = (w >= min_contour_width) and (
         #       h >= min_contour_height)

        #if not contour_valid:
           # continue



    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w + 5, y + h + 5), (0, 0, 255), 3)
        #cv2.line(img, (0, line_height), (1200, line_height), (0, 255, 0), 2)
        cX = int((x + x + w) / 2.0)
        cY = int((y + y + h) / 2.0)
        #cv2.circle(img, (cX, cY), 4, (255, 0, 0), -1)
        centroid = get_centroid(x, y, w, h)
        matches.append(centroid)
        cv2.circle(img, centroid, 5, (0, 255, 0), -1)
        cx,cy = get_centroid(x, y, w, h)
        for (x,y) in matches:
            if y < (line_height+50) and y > (line_height-50):
              count=count+1
              matches.remove((x, y))
              print(cars)


    cv2.putText(img, "Total Cars Detected: " + str(count), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 190), 2)

    cv2.putText(img, "SDL MINI PROJECT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 170, 0), 2)

    #roi_gray = gray[y:y+h,x:x+w]
        #roi_color=img[y:y+h,x:x+w]
        #count+=1

    cv2.imshow('Car Detector', img)
    #count = count + len(cars)

   #print(type(cars), cars)
    k = cv2.waitKey(30) & 0xff
    if k == ord("s"):

        initBB = cv2.selectROI("Frame", img, fromCenter=False,
                               showCrosshair=True)


        tracker.init(img, initBB)
        fps = FPS().start()


    elif k == ord("q"):
     if k == 27:
      break

cap.release()
cv2.destroyAllWindows()


