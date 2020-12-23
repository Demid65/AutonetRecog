import cv2
import time
from threading import Thread
import globalDefs

class lineThread(Thread):
    def __init__(self, im_width = 640, im_height = 480):
        Thread.__init__(self)
        print('Line thread: initialized')

    def run(self):
        while globalDefs.aliveFlag:
            flag, frame = globalDefs.lineCam.read()
            img = frame[100:220, 100:220]
            globalDefs.lineFrameCropped = img
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
            globalDefs.lineFrameThresh = thresh
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
                globalDefs.lineVal = 80 - cx
            time.sleep(0.01)







