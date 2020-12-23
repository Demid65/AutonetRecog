import cv2
import numpy as np
cap = cv2.VideoCapture(1)
cap.set(3, 320)
cap.set(4, 240)

speed = 115.0
kp = 1.3
while True:
    flag, frame = cap.read()
    img = frame[100:220, 100:220]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh.copy(), 1, cv2.CHAIN_APPROX_NONE)
    res = 0
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        m00 = M['m00']
        m10 = M['m10']
        if m00 != 0:
            cx = int(m10 / m00)
        else:
            cx = 80
        res = 80 - cx