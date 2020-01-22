# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import serial
import socket
import time
import sys
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util







# Name of the directory containing the object detection module we're using
MODEL_NAME = 'models'


# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,'labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 10

def avg_hue(img):
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    r=np.take(avg_color,2)
    g=np.take(avg_color,1)
    b=np.take(avg_color,0)
    v=max(r,g,b)
    m=min(r,g,b)
    #print(r,g,b)
    if v==r and g>=b:
        h=60*(g-b)/(v-m)
    elif v==r and g<b:
        h=60*(g-b)/(v-m)+360
    elif v==g:
        h=60*(b-r)/(v-m)+120
    elif v==b:
        h=60*(r-g)/(v-m)+240
    return h

def hue2cid(hue):     #0r 1b 2g 3y 4o
    #print(hue)
    if hue>300:
        cid=0
    elif hue<20: 
        cid=0
    elif hue>180 and hue<300:
        cid=1
    elif hue>60 and hue<180:	
        cid=2
    elif hue>35 and hue<60:		
        cid=3
    elif hue>20 and hue<35:		
        cid=4
    return cid

def Recog(sending):
    print('starting recog')
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    if np.take(scores,0) >0.5:
         card_id=int(np.take(classes,0))
         ymin=np.take(boxes,0)
         xmin=np.take(boxes,1)
         ymax=np.take(boxes,2)
         xmax=np.take(boxes,3)
         (xminn, xmaxx, yminn, ymaxx) = (int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))
         cropped_image = frame[yminn:ymaxx, xminn:xmaxx]
         cv2.imshow('Cropped',cropped_image)
         color_id=hue2cid(avg_hue(cropped_image))
         
         #print(avg_hue(cropped_image))
         if card_id==10:
             card_id2send=0
         else:
             card_id2send=card_id
         if color_id==0:
            #sys.stdout.write(str(card_id2send)+'r')
            if sending:
                conn.send((str(card_id2send)+'r').encode())
            if SM:
                ser.write('(1)[51]<DigitalWrite>'.encode())
                ser.write('(0)[52]<DigitalWrite>'.encode())
                ser.write('(0)[53]<DigitalWrite>'.encode())
                ser.write('(0)[13]<DigitalWrite>'.encode())

			
         elif color_id==1:
            #sys.stdout.write(str(card_id2send)+'b')
            if sending:
                conn.send((str(card_id2send)+'b').encode())
            if SM:
                ser.write('(0)[51]<DigitalWrite>'.encode())
                ser.write('(1)[52]<DigitalWrite>'.encode())
                ser.write('(0)[53]<DigitalWrite>'.encode())
                ser.write('(0)[13]<DigitalWrite>'.encode())

         elif color_id==2:
            #sys.stdout.write(str(card_id2send)+'g')
            if sending:
                conn.send((str(card_id2send)+'g').encode())
            if SM:
                ser.write('(0)[51]<DigitalWrite>'.encode())
                ser.write('(0)[52]<DigitalWrite>'.encode())
                ser.write('(1)[53]<DigitalWrite>'.encode())
                ser.write('(0)[13]<DigitalWrite>'.encode())
         elif color_id==3:
            #sys.stdout.write(str(card_id2send)+'y')
            if sending:
                conn.send((str(card_id2send)+'y').encode())
            if SM:
                ser.write('(1)[51]<DigitalWrite>'.encode())
                ser.write('(1)[52]<DigitalWrite>'.encode())
                ser.write('(0)[53]<DigitalWrite>'.encode())
                ser.write('(0)[13]<DigitalWrite>'.encode())
         elif color_id==4:
            if sending:
                conn.send((str(card_id2send)+'o').encode())    
    else:
        print('Empty')
        if sending:
            conn.send('Empty'.encode())

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)

    
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)	


def CurrFrame():
    ret, frame = video.read()
    cv2.imshow('current frame',frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        Shutdown()
    if key==ord(' '):
        Recog(False)
    if key==ord('c'):
        print(str(hue2cid(avg_hue(frame))))
    return frame

def Shutdown():
    cv2.destroyAllWindows()
    video.release()
    if SM:
        ser.close()
    exit()

def ReconnectLoop():
    print('waiting for connection')
    isConn=False
    while not isConn:
        try:
            conn, addr=sock.accept()
            isConn=True
        except socket.timeout:
            a=1
        CurrFrame()
    print('connection from ',addr)
    conn.settimeout(0.1)
    return conn
    


## Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')



# Initialize webcam feed
video = cv2.VideoCapture(0)
# ret = video.set(3,1280)
# ret = video.set(4,720)
video.set(cv2.CAP_PROP_BUFFERSIZE, 1);

ret = video.set(3,640)
ret = video.set(4,320)
im_height=320
im_width=640

a='a'
SM=False
sock = socket.socket()
sock.bind(('',9090))
sock.listen(1)
sock.settimeout(0.1)
conn=ReconnectLoop()
#conn.setblocking(False)
if SM:
    ser=serial.Serial('COM31')
    ser.write('(0)[51]<PinMode>'.encode())
    ser.write('[52]<PinMode>'.encode())
    ser.write('[53]<PinMode>'.encode())
    ser.write('[13]<PinMode>'.encode())
    ser.write('(1)[13]<DigitalWrite>'.encode())

while(True):
    c = cv2.waitKey(1)
    frame = CurrFrame()
    c = ''
    try:
        c = conn.recv(64)
    except socket.timeout:
	    conn = ReconnectLoop()
    except ConnectionResetError:
        print('Connection lost')
        cv2.destroyAllWindows()
        conn = ReconnectLoop()
        #print('timeout') 				
    if c == b' ':
        Recog(True)
    if c == b'c':
        conn.send(str(hue2cid(avg_hue(frame))).encode())

    
 
 
		
