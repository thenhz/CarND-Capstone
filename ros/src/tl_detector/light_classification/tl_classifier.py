from styx_msgs.msg import TrafficLight

import tensorflow as tf
from keras.models import load_model
from keras.utils.data_utils import get_file
import os
import numpy as np
import cv2
import errno

class TLClassifier(object):
    def __init__(self, **kwargs):

        self.path_detector = "../../../models/frozen_inference_graph.pb"
        self.path_classifier = "../../../models/model.h5"
        self.detector = Detector(self.path_detector)
        self.classifier = Classifier(self.path_classifier)

    def get_classification(self, image):

        (boxes, scores, classes, num_detections) = self.detector.detect(image)
        tl_detections = [i for i in range(boxes.shape[1]) if (scores[0, i] > 0.8 and classes[0, i] == 10)]
        if len(tl_detections) == 0:
            print ("No traffic light detected")
            return TrafficLight.UNKNOWN
        else:
            print ("Detected possible {} traffic lights".format(len(tl_detections)))

        cropped = np.array(filter(lambda x: x is not None, [self._prepare_for_class(image, boxes[:, i, :]) for i in tl_detections if i is not None]))
        
        if len(cropped) == 0:
            print ("Detected no traffic lights...")
            return TrafficLight.UNKNOWN

        classification = self.classifier.classify(cropped)
        return TLClassifier.eval_color(classification)

    @staticmethod
    def eval_color(classification):
        if classification == 0:
            return TrafficLight.RED
        elif classification == 1:
            return TrafficLight.YELLOW
        elif classification == 2:
            return TrafficLight.GREEN
        else:
            return TrafficLight.UNKNOWN

    @staticmethod
    def _prepare_for_class(image, box):
        """
        It crops a box from the image
        :param image: The image
        :param box: The box to crop
        :return: The resized cropped image or None if the conditions are not satisfied
        """
        shape = image.shape
        print("shape:",shape)
        (left, right, top, bottom) = (box[0, 1] * shape[2], box[0, 3] * shape[2],
                                      box[0, 0] * shape[1], box[0, 2] * shape[1])

        # Assuming that crop_height > crop_width valid for tf_api and standard traffic lights
        crop_height = int(bottom - right)
        crop_width = int(top - left)
        print(left,right,top, bottom)
        cropped = image[int(top):int(top)+crop_height, int(left):int(left)+crop_width]
        resized = cv2.resize(cropped, (50, 50), interpolation=cv2.INTER_CUBIC)
        return resized[..., ::-1]
        


class Detector(object):
    def __init__(self, model_file):

        self.model_file = model_file
        self.detection_graph = tf.Graph()
        self._import_tf_graph()
        self.sess = tf.Session(graph=self.detection_graph)

    def _import_tf_graph(self):
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def detect(self, image_np):
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        return self.sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})


class Classifier(object):
    def __init__(self, model_path):

        self.classification_model = load_model(model_path)

        self.classification_model.summary()

        self.classification_model._make_predict_function()  # see https://github.com/fchollet/keras/issues/6124
        self.classification_graph = tf.get_default_graph()

    def classify(self, cropped):
        with self.classification_graph.as_default():
            predictions = self.classification_model.predict(cropped)

        results = []
        print( predictions)
        for i,p in enumerate(predictions):
            if np.max(p) > 0.9:
                result = np.argmax(p)
                results.append(result)
        if len(results) == 0:
            return None
        else:
            counts = np.bincount(results)
            if len(counts[counts == np.max(counts)]) == 1:
                return np.argmax(counts)
            else:
                return None