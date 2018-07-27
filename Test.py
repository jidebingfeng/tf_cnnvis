from tf_cnnvis.tf_cnnvis import *
import os
import matplotlib.image as mpimg

import tensorflow as tf
print(tf.__version__)


PATH_TO_CKPT = 'data/tf1.9.5819/frozen_inference_graph.pb'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
input_tensor_placehodler = detection_graph.get_tensor_by_name('FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/Pad:0')
image = mpimg.imread('data/1.jpg')
image_np_expanded = np.expand_dims(image, axis=0)


layers = ['SecondStageFeatureExtractor/resnet_v1_101/block4/unit_1/bottleneck_v1/shortcut/Conv2D',]


is_success = deconv_visualization(sess_graph_path='data/tf1.9.5819/model.ckpt.meta', value_feed_dict={image_tensor: image_np_expanded},
                                  input_tensor=input_tensor_placehodler,
                                  layers=layers, path_logdir=os.path.join("Log", "Inception5"),
                                  path_outdir=os.path.join("Log", "Inception5"))
