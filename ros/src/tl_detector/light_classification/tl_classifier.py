from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
import rospy
import datetime
import numpy as np
import os
import time


class TLClassifier(object):
    def __init__(self):

	base_folder = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join('', '../../../models/final/frozen_inference_graph.pb')

	self.graph = tf.Graph()

	with self.graph.as_default():
	    od_graph_def = tf.GraphDef()
	    with tf.gfile.GFile(model_path, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

	self.sess = tf.Session(graph = self.graph)
	self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

	self.classes = {
            1: TrafficLight.GREEN,
            2: TrafficLight.YELLOW,
            3: TrafficLight.RED
        }

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
	image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
    	(boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], feed_dict={self.image_tensor: image_np})

    	# Remove unnecessary dimensions
    	boxes = np.squeeze(boxes)
    	scores = np.squeeze(scores)
    	classes = np.squeeze(classes)

    	confidence_cutoff = 0.5

	n = len(classes)
	
	for i in range(n):	
	    if scores[i] >= confidence_cutoff:
		rospy.logwarn("tl_classifier: Traffic Light Class detected: %d with score %f", classes[i], scores[i])
		return self.classes[classes[i]]

        return TrafficLight.UNKNOWN