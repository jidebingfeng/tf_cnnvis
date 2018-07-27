from tf_cnnvis.tf_cnnvis import *
import os
import matplotlib.image as mpimg

import tensorflow as tf
print(tf.__version__)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_dir = "/mnt/lost+found/data/gen/jinju.classic/data/faster_rcnn_resnet101/80k.tf9"
log_base_dir = "/mnt/lost+found/bigdata_workspace/docker/tf_cnnvis/log/"

path_pb = os.path.join(model_dir, 'frozen_inference_graph.pb')
graph_path = os.path.join(model_dir, 'model.ckpt.meta')
path_log = os.path.join(log_base_dir, 'log')
path_out = os.path.join(log_base_dir, 'output')

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_pb, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
input_tensor_placehodler = detection_graph.get_tensor_by_name('FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/Pad:0')
image = mpimg.imread('/mnt/lost+found/data/gen/image.scale.fix.736.491/images/18062811215.jpg')
image_np_expanded = np.expand_dims(image, axis=0)


all_layers = ['FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block1/unit_1/bottleneck_v1/conv1/Conv2D',]
all_layers= []
for op in detection_graph.get_operations():
    t = op.type.lower()
    if t == 'maxpool' or t == '' or t == 'relu' or t == 'conv2d':
        if op.name.find('FirstStageFeatureExtractor') > -1:
            all_layers.append(op.name)


layers = []
for index,layer in enumerate (all_layers):
    layer_dir_name =layer.replace('/', '_').lower()
    layer_dir = os.path.join(path_log,layer_dir_name)
    if os.path.exists(layer_dir):
        print("Exists:",layer_dir)
    else :
        layers.append(layer)

is_success = deconv_visualization(sess_graph_path=graph_path, value_feed_dict={image_tensor: image_np_expanded},
                                  input_tensor=input_tensor_placehodler,
                                  layers=layers, path_logdir=path_log,
                                  path_outdir=path_out)


