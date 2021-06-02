import tensorflow as tf
import cfg
import loss
from net import resnet
from net.layers import _conv, upsampling, _conv_nn
import numpy as np


class CenterNet():
    def __init__(self, inputs, is_training):
        self.is_training = is_training
        try:
            self.pred_hm, self.pred_wh, self.pred_reg = self._build_model(inputs)
        except:
            raise NotImplementedError("Can not build up centernet network!")

    def _build_model(self, inputs):
        with tf.variable_scope('resnet'):
            c2, c3, c4, c5 = resnet.resnet18(is_training=self.is_training).forward(inputs)

            p5 = _conv(c5, 128, [1,1], is_training=self.is_training, name="conv_p5")
            up_p5 = upsampling(p5)

            reduce_dim_c4 = _conv(c4, 128, [1,1], is_training=self.is_training, name="conv_c4")
            p4 = 0.5*up_p5 + 0.5*reduce_dim_c4
            up_p4 = upsampling(p4)
            
            reduce_dim_c3 = _conv(c3, 128, [1,1], is_training=self.is_training, name="conv_c3")
            p3 = 0.5*up_p4 + 0.5*reduce_dim_c3
            up_p3 = upsampling(p3)

            reduce_dim_c2 = _conv(c2, 128, [1,1], is_training=self.is_training, name="conv_c2")
            p2 = 0.5*up_p3 + 0.5*reduce_dim_c2
            features = _conv(p2, 128, [3,3], is_training=self.is_training, name="conv_p2")

            # IDA-up
            # p2 = _conv(c2, 128, [1,1], is_training=self.is_training)
            # p3 = _conv(c3, 128, [1,1], is_training=self.is_training)
            # p4 = _conv(c4, 128, [1,1], is_training=self.is_training)
            # p5 = _conv(c5, 128, [1,1], is_training=self.is_training)

            # up_p3 = upsampling(p3, method='resize')
            # p2 = _conv(p2+up_p3, 128, [3,3], is_training=self.is_training)

            # up_p4 = upsampling(upsampling(p4, method='resize'), method='resize')
            # p2 = _conv(p2+up_p4, 128, [3,3], is_training=self.is_training)

            # up_p5 = upsampling(upsampling(upsampling(p5, method='resize'), method='resize'), method='resize')
            # features = _conv(p2+up_p5, 128, [3,3], is_training=self.is_training)
        
        with tf.variable_scope('detector'):
            print('feature shape: ', features.shape)
            hm = _conv(features, 64, [3,3], is_training=self.is_training, name="hm_conv_1")
            hm = _conv_nn(hm, cfg.num_classes, [1, 1], padding='VALID', activation = tf.nn.relu, name='hm_conv')
            #hm = _conv(hm, cfg.num_classes, [1, 1], padding="valid", name="hm")
        

            wh = _conv(features, 64, [3,3], is_training=self.is_training, name="wh_conv_1")
            #wh = tf.layers.conv2d(wh, 2, 1, 1, padding='valid', activation = None, bias_initializer=tf.constant_initializer(-np.log(99.)), name='wh')
            wh = _conv_nn(wh, 2, [1, 1], padding='VALID', activation = tf.nn.relu, name="wh_conv")
            #wh = tf.reshape(wh, [-1, wh.shape[1], wh.shape[2], wh.shape[3]])
            #wh = tf.layers.conv2d(wh, 2, 1, 1, padding='valid', activation = None, name='wh')
            #wh = _conv(wh, 2, [1, 1], padding="valid", name="wh")
        

            reg =  _conv(features, 64, [3,3], is_training=self.is_training, name="reg_conv_1")
            #reg = tf.layers.conv2d(reg, 2, 1, 1, padding='valid', activation = None, bias_initializer=tf.constant_initializer(-np.log(99.)), name='reg')
            reg = _conv_nn(reg, 2, [1, 1], padding='VALID', activation = tf.nn.relu, name="reg_conv")
            #reg = tf.reshape(reg, [-1, reg.shape[1], reg.shape[2], reg.shape[3]])
            #reg = _conv(reg, 2, [1, 1], padding="valid", name="reg")
     

        return hm, wh, reg

    def compute_loss(self, true_hm, true_wh, true_reg, reg_mask, ind):
        hm_loss = loss.focal_loss(self.pred_hm, true_hm)
        wg_loss = 0.05*loss.reg_l1_loss(self.pred_wh, true_wh, ind, reg_mask)
        reg_loss = loss.reg_l1_loss(self.pred_reg, true_reg, ind, reg_mask)
        return hm_loss, wg_loss, reg_loss