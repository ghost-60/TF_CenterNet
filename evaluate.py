#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : evaluate.py
#   Author      : YunYang1994
#   Created date: 2019-02-21 15:30:26
#   Description :
#
#================================================================

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import utils.utils as utils
import cfg
from CenterNet import CenterNet
from utils.decode import decode
import xml.etree.ElementTree as gfg  
from xml.dom import minidom

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class CenternetTest(object):
    def __init__(self):
        self.input_size       = cfg.input_image_h
        self.classes          = utils.read_class_names(cfg.classes_file)
        self.num_classes      = len(self.classes)
        self.score_threshold  = cfg.score_threshold
        self.iou_threshold    = cfg.nms_thresh
        self.moving_ave_decay = cfg.moving_ave_decay
        self.annotation_path  = cfg.test_data_file
        self.weight_file      = cfg.weight_file
        self.write_image      = cfg.write_image
        self.write_image_path = cfg.write_image_path
        self.show_label       = cfg.show_label

        self.input_data = tf.placeholder(shape=[1,cfg.input_image_h, cfg.input_image_w,3],dtype=tf.float32)
        
        self.sess  = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        self.sess.run(tf.global_variables_initializer())
        
        model = CenterNet(self.input_data, False)
        saver = tf.train.Saver()
        saver.restore(self.sess,'./checkpoint/2021_02_24-centernet_test_loss=0.5797.ckpt-80')

        self.hm = model.pred_hm
        self.wh = model.pred_wh
        self.reg = model.pred_reg

        self.det = decode(self.hm, self.wh, self.reg, K=cfg.max_objs)

    def predict(self, image):

        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape
        original_image_size = org_image.shape[:2]

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        detections = self.sess.run(self.det, feed_dict={self.input_data: image_data})
        detections = utils.post_process(detections, original_image_size, [cfg.input_image_h,cfg.input_image_w], cfg.down_ratio, cfg.score_threshold)

        bboxes = []
        scores = [0]
        classes = [0]
        if cfg.use_nms:
            cls_in_img = list(set(detections[:,5]))
            results = []
            for c in cls_in_img:
                cls_mask = (detections[:,5] == c)
                classified_det = detections[cls_mask]
                classified_bboxes = classified_det[:, :4]
                classified_scores = classified_det[:, 4]
                inds = utils.py_nms(classified_bboxes, classified_scores, max_boxes=50, iou_thresh=0.5)
                results.extend(classified_det[inds])
            results = np.asarray(results)
            if len(results) != 0:
                bboxes = results[:,0:4]
                scores = results[:,4]
                classes = results[:, 5]
                #bboxes_draw_on_img(original_image, classes, scores, bboxes, class_names)
        else:
            bboxes = detections[:,0:4]
            scores = detections[:,4]
            classes = detections[:,5]
        #bboxes_draw_on_img(original_image, classes, scores, bboxes, class_names)

        return bboxes, scores, classes

    def evaluate(self):
        predicted_dir_path = './mAP/predicted'
        ground_truth_dir_path = './mAP/ground-truth'
        if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(self.write_image_path): shutil.rmtree(self.write_image_path)
        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)
        os.mkdir(self.write_image_path)

        with open(self.annotation_path, 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('\\')[-1]
                image = cv2.imread(image_path)
                bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

                if len(bbox_data_gt) == 0:
                    bboxes_gt=[]
                    classes_gt=[]
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')
                self.saveXML(bboxes_gt, image_name, image.shape, "eval/Annotations_ground/")
                print('=> ground truth of %s:' % image_name)
                num_bbox_gt = len(bboxes_gt)
                with open(ground_truth_path, 'w') as f:
                    for i in range(num_bbox_gt):
                        class_name = self.classes[classes_gt[i]]
                        xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                        bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())
                print('=> predict result of %s:' % image_name)
                predict_result_path = os.path.join(predicted_dir_path, image_name[:-4] + '.txt')
                bboxes_pr, scores_pr, classes_pr = self.predict(image)

                if self.write_image:
                    #image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
                    cv2.imwrite(self.write_image_path+image_name, image)

                with open(predict_result_path, 'w') as f:
                    for bbox in bboxes_pr:
                        coor = np.array(bbox[:4], dtype=np.int32)
                        score = scores_pr
                        class_ind = classes_pr
                        class_name = self.classes[0]
                        #print(class_name, class_ind)
                        
                        score = '%.4f' % max(score)
                        xmin, ymin, xmax, ymax = list(map(str, coor))
                        bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())
                self.saveXML(bboxes_pr, image_name, image.shape, "eval/Annotations/")

    def voc_2012_test(self, voc2012_test_path):

        img_inds_file = os.path.join(voc2012_test_path, 'ImageSets', 'Main', 'test.txt')
        with open(img_inds_file, 'r') as f:
            txt = f.readlines()
            image_inds = [line.strip() for line in txt]

        results_path = 'results/VOC2012/Main'
        if os.path.exists(results_path):
            shutil.rmtree(results_path)
        os.makedirs(results_path)

        for image_ind in image_inds:
            image_path = os.path.join(voc2012_test_path, 'JPEGImages', image_ind + '.jpg')
            image = cv2.imread(image_path)

            print('predict result of %s:' % image_ind)
            bboxes_pr, scores_pr, classes_pr = self.predict(image)
            for bbox in bboxes_pr:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = scores_pr
                class_ind = classes_pr
                class_name = self.classes[0]
                score = '%.4f' % max(score)
                xmin, ymin, xmax, ymax = list(map(str, coor))
                bbox_mess = ' '.join([image_ind, score, xmin, ymin, xmax, ymax]) + '\n'
                with open(os.path.join(results_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
                    f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())
            self.saveXML(bboxes_pr, image_ind, image.shape, "eval/Annotations/")

    
    def saveXML(self, bboxes, filename, shape, save_path):
        [l, w, d] = shape
        
        root = gfg.Element("annotation") 
            
        m1 = gfg.Element("filename") 
        m1.text = filename
        root.append (m1) 

        m2 = gfg.Element('size')
        root.append(m2)

        b1 = gfg.SubElement(m2, "width") 
        b1.text = str(w)
        b2 = gfg.SubElement(m2, "height") 
        b2.text = str(l)
        b2 = gfg.SubElement(m2, "depth") 
        b2.text = str(d)
        for bbox in bboxes:
            m3 = gfg.Element("object") 
            root.append (m3) 
                
            c1 = gfg.SubElement(m3, "name") 
            c1.text = "drone"

            c2 = gfg.SubElement(m3, "bndbox") 
            d1 = gfg.SubElement(c2, "xmin")
            d1.text = str(bbox[0])
            d2 = gfg.SubElement(c2, "ymin")
            d2.text = str(bbox[1])
            d3 = gfg.SubElement(c2, "xmax")
            d3.text = str(bbox[2])
            d4 = gfg.SubElement(c2, "ymax")
            d4.text = str(bbox[3])

        tree = gfg.ElementTree(root) 
        xmlstr = minidom.parseString(gfg.tostring(root)).toprettyxml(indent="   ")
    
        with open (save_path + filename[:-4] + ".xml", "w") as files : 
            files.write(xmlstr)


if __name__ == '__main__': CenternetTest().evaluate()



