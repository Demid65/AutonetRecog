import os
import cv2
import rgbTools
import numpy as np
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
from threading import Thread
import globalDefs

class recogThread(Thread):
    def __init__(self, im_width = 640, im_height = 480):
        Thread.__init__(self)
        # Name of the directory containing the object detection module we're using
        MODEL_NAME = 'models'
        # Grab path to current working directory
        CWD_PATH = os.getcwd()
        # Path to frozen detection graph .pb file, which contains the model that is used for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, 'labelmap.pbtxt')
        # Number of classes the object detector can identify
        NUM_CLASSES = 10

        self.im_width = 640
        self.im_height = 480

        # Load the label map.
        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        # Load the Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        print('Recog thread: initialized')

    def run(self):
        while globalDefs.aliveFlag:
            ret, frame = globalDefs.recogCam.read()
            frame_expanded = np.expand_dims(frame, axis=0)

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: frame_expanded})

            if np.take(scores, 0) > 0.9:
                globalDefs.cardNum = int(np.take(classes, 0))
                ymin = np.take(boxes, 0)
                xmin = np.take(boxes, 1)
                ymax = np.take(boxes, 2)
                xmax = np.take(boxes, 3)
                (xminn, xmaxx, yminn, ymaxx) = (
                    int(xmin * self.im_width), int(xmax * self.im_width), int(ymin * self.im_height), int(ymax * self.im_height))
                cropped_image = frame[yminn:ymaxx, xminn:xmaxx]
                globalDefs.CardCID = rgbTools.hue2cid(rgbTools.avg_hue(cropped_image))
            else:
                globalDefs.cardNum = None

            # Draw the results of the detection
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.9)

            # All the results have been drawn on the frame, so it's time to display it.
            globalDefs.recogFrame = frame
