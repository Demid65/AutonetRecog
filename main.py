# Import packages
import cv2
import recogThread
import lineThread
import serverThread
import globalDefs

def UpdateFrame():
    ret, frame = globalDefs.recogCam.read()
    if (ret):
        cv2.imshow('recogCam', frame)
    ret, frame = globalDefs.lineCam.read()
    if (ret):
        cv2.imshow('lineCam', frame)
    cv2.imshow('recogFrame', globalDefs.recogFrame)
    cv2.imshow('lineFrameCropped', globalDefs.lineFrameCropped)
    cv2.imshow('lineFrameThresh', globalDefs.lineFrameThresh)
    key = cv2.waitKey(1)
    if key == ord('q'):
        Shutdown()
    return frame


def Shutdown():
    print('Main thread: Shutting down')
    globalDefs.aliveFlag = False
    RecogThread.join()
    print('Main thread: Recog thread was terminated')
    LineThread.join()
    print('Main thread: Line thread was terminated')
    ServerThread.join()
    print('Main thread: Server thread was terminated')
    cv2.destroyAllWindows()
    globalDefs.recogCam.release()
    exit()

# Initialize webcam feed
globalDefs.init()
globalDefs.recogCam = cv2.VideoCapture(0)
globalDefs.recogCam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

globalDefs.lineCam = cv2.VideoCapture(1)
globalDefs.lineCam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

ret = globalDefs.recogCam.set(3, 640)
ret = globalDefs.recogCam.set(4, 480)

ret = globalDefs.lineCam.set(3, 320)
ret = globalDefs.lineCam.set(4, 240)

ret, globalDefs.recogFrame = globalDefs.recogCam.read()
ret, globalDefs.lineFrameCropped = globalDefs.lineCam.read()
ret, globalDefs.lineFrameThresh = globalDefs.lineCam.read()

print('Main thread: initialized')

RecogThread = recogThread.recogThread()
RecogThread.start()

LineThread = lineThread.lineThread()
LineThread.start()

ServerThread = serverThread.serverThread()
ServerThread.start()

while (True):
    UpdateFrame()