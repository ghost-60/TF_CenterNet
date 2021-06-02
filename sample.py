import numpy as np
import tensorflow as tf
from utils.utils import image_preporcess
import os
import random
import cv2
import cfg
from tensorflow.python.platform import gfile
from tensorflow.python.framework.graph_util import convert_variables_to_constants

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

graph = tf.Graph()

with tf.gfile.GFile('checkpoint/centernet_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

print('Check out the input placeholders:')
nodes = [
    n.name + ' => ' + n.op for n in graph_def.node
    if n.op in ('Placeholder')
]
for node in nodes:
    print(node)

with graph.as_default():
    inputs = tf.placeholder(np.float32, shape = [1, 128, 128, 3], name='input')

    tf.import_graph_def(graph_def, {
        'Placeholder:0': inputs
    })

graph.finalize()

print('Model loading complete!')

# Get layer names
layers = [op.name for op in graph.get_operations()]
for layer in layers:
    print(layer)
"""
# Check out the weights of the nodes
weight_nodes = [n for n in graph_def.node if n.op == 'Const']
for n in weight_nodes:
    print("Name of the node - %s" % n.name)
    # print("Value - " )
    # print(tensor_util.MakeNdarray(n.attr['value'].tensor))
"""


sess = tf.Session(graph=graph)

img_names = os.listdir('D:\\Courses\\Yolo\\DroneDataset\\dataset_new\\JPEGImages\\')
random.shuffle(img_names)
for img_name in img_names:
    img_path = 'D:\\Courses\\Yolo\\DroneDataset\\dataset_new\\JPEGImages\\' + img_name
    original_image = cv2.imread(img_path)
    print(original_image)
    original_image_size = original_image.shape[:2]
    image_data = image_preporcess(np.copy(original_image), [cfg.input_image_h, cfg.input_image_w])
    image_data = image_data[np.newaxis, ...]
    print(image_data)
    print('shape: ', image_data.shape)
    output_tensor = graph.get_tensor_by_name("import/detector/Relu_1:0")
    output = sess.run(output_tensor, feed_dict = {inputs: image_data})
    print(output[output != 0])
    break
