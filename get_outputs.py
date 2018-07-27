from tf_cnnvis.tf_cnnvis import *
import os
import matplotlib.image as mpimg
from scipy.misc import imread, imresize
import collections


import tensorflow as tf
print(tf.__version__)




# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'data/tf1.9.5819/frozen_inference_graph.pb'

NUM_CLASSES = 9

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


print("=====================Operations==============================")
for op in detection_graph.get_operations():
    print(op.name,op.type)

print("=====================Tensors==============================")
layers = []
for op in detection_graph.get_operations():
    t = op.type.lower()
    if t == 'maxpool' or t == '' or t == 'relu' or t == 'conv2d':
        if op.name.find('Second') > -1:
            layers.append(op.name)
            print(op.name)

            for input in op.inputs:
                print(input.shape, input.dtype)


print("=========================================================")
print("=========================================================")
print("=====================Layers==============================")
print("=========================================================")
print("=========================================================")
print('Number of layers', len(layers))


image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
image = mpimg.imread('data/1.jpg')
image_np_expanded = np.expand_dims(image, axis=0)

print("===================================================")
layer_name = 'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/conv1/Conv2D'
# layer_name = 'FirstStageFeatureExtractor/resnet_v1_101/conv1/weights/read'
# layer_name = 'FirstStageFeatureExtractor/resnet_v1_101/conv1/weights'
# layer_name = 'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/Pad'
# layer_name = 'Preprocessor/sub'
op = detection_graph.get_operation_by_name(layer_name)
for input in op.inputs:
    print(input.name,input.shape,input.dtype)

print("Success!")

