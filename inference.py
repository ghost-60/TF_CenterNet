import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import time
import cfg
from CenterNet import CenterNet
from utils.decode import decode
from utils.utils import image_preporcess, py_nms, post_process, bboxes_draw_on_img, read_class_names
import random


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

ckpt_path='./checkpoint/'
sess = tf.Session()

inputs = tf.placeholder(shape=[1,cfg.input_image_h, cfg.input_image_w,3],dtype=tf.float32)
model = CenterNet(inputs, False)
saver = tf.train.Saver()
saver.restore(sess,tf.train.latest_checkpoint(ckpt_path))

hm = model.pred_hm
wh = model.pred_wh
reg = model.pred_reg
# c2 = model.c2
# c3 = model.c3
# c4 = model.c4
# c5 = model.c5
# first_layer = model.first_layer
det = decode(hm, wh, reg, K=cfg.max_objs)

class_names= read_class_names(cfg.classes_file)
img_names = os.listdir('D:\\Courses\\Yolo\\DroneDataset\\dataset_new\\JPEGImages\\')
random.shuffle(img_names)

#ot_nodes = ['detector/hm/Sigmoid', "detector/wh/BiasAdd", "detector/reg/BiasAdd"]
ot_nodes = cfg.ot_nodes
#ot_nodes = ['detector/Conv2D_1', 'detector/Conv2D_3', 'detector/Conv2D_5']
writer = tf.compat.v1.summary.FileWriter("./output", sess.graph)
for img_name in img_names:
    img_path = 'D:\\Courses\\Yolo\\DroneDataset\\dataset_new\\JPEGImages\\' + img_name
    original_image = cv2.imread(img_path)
    original_image_size = original_image.shape[:2]
    image_data = image_preporcess(np.copy(original_image), [cfg.input_image_h, cfg.input_image_w])
    image_data = image_data[np.newaxis, ...]
    print('shape: ', image_data.shape)
    t0 = time.time()
    
    frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess,
                sess.graph_def,
                ot_nodes)
    with open('checkpoint/centernet_graph.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())
    #saver2  = tf.train.Saver(tf.global_variables(), max_to_keep=10)
    #saver2.save(sess, "checkpoint\\inference\\model_inference.ckpt")
    writer.close()
    detections = sess.run(det, feed_dict={inputs: image_data})
    # heat = sess.run(first_layer, feed_dict={inputs: image_data})
    # print(heat)
    #print(detections[0, :, 4])
    detections = post_process(detections, original_image_size, [cfg.input_image_h,cfg.input_image_w], cfg.down_ratio, cfg.score_threshold)
    
    print('Inferencce took %.1f ms (%.2f fps)' % ((time.time()-t0)*1000, 1/(time.time()-t0)))
    if cfg.use_nms:
        cls_in_img = list(set(detections[:,5]))
        results = []
        for c in cls_in_img:
            cls_mask = (detections[:,5] == c)
            classified_det = detections[cls_mask]
            classified_bboxes = classified_det[:, :4]
            classified_scores = classified_det[:, 4]
            inds = py_nms(classified_bboxes, classified_scores, max_boxes=50, iou_thresh=0.5)
            results.extend(classified_det[inds])
        results = np.asarray(results)
        if len(results) != 0:
            bboxes = results[:,0:4]
            scores = results[:,4]
            classes = results[:, 5]
            bboxes_draw_on_img(original_image, classes, scores, bboxes, class_names)
        
    else:
        bboxes = detections[:,0:4]
        scores = detections[:,4]
        classes = detections[:,5]
        bboxes_draw_on_img(original_image, classes, scores, bboxes, class_names)

    cv2.imshow('img',original_image)
    cv2.waitKey()
