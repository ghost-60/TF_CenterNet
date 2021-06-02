import os
import numpy as np
import tensorflow as tf
import cv2
import math
import time
import shutil
import cfg
from tqdm import tqdm
from CenterNet import CenterNet
from utils.generator import get_data
from net.resnet import load_weights
from datetime import datetime

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def getVal(temp_input):
    return temp_input

def train():
    # define dataset
    num_train_imgs = len(open(cfg.train_data_file, 'r').readlines())
    num_train_batch = int(math.ceil(float(num_train_imgs) / cfg.batch_size))
    num_test_imgs = len(open(cfg.test_data_file, 'r').readlines())
    num_test_batch = int(math.ceil(float(num_test_imgs) / 2))

    train_dataset = tf.data.TextLineDataset(cfg.train_data_file)
    train_dataset = train_dataset.shuffle(num_train_imgs)
    train_dataset = train_dataset.batch(cfg.batch_size)
    train_dataset = train_dataset.map(lambda x: tf.py_func(get_data,inp=[x, True], 
                                    Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]),
                                    num_parallel_calls=7)
    train_dataset = train_dataset.prefetch(3)
    
    test_dataset = tf.data.TextLineDataset(cfg.test_data_file)
    test_dataset = test_dataset.batch(2)
    test_dataset = test_dataset.map(lambda x: tf.py_func(get_data,inp=[x, False], 
                                    Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]),
                                    num_parallel_calls=7)
    test_dataset = test_dataset.prefetch(3)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    trainset_init_op = iterator.make_initializer(train_dataset)
    testset_init_op = iterator.make_initializer(test_dataset)

    input_data, batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_ind, batch_pose = iterator.get_next()
    input_data.set_shape([None, cfg.input_image_h, cfg.input_image_w, 3])
    batch_hm.set_shape([None,cfg.input_image_h//cfg.down_ratio, cfg.input_image_w//cfg.down_ratio, cfg.num_classes])
    batch_wh.set_shape([None, cfg.max_objs, 2])
    batch_reg.set_shape([None, cfg.max_objs, 2])
    batch_reg_mask.set_shape([None, cfg.max_objs])
    batch_ind.set_shape([None, cfg.max_objs])
    batch_pose.set_shape([None, cfg.max_objs, cfg.depth])


    # training flag 
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')
    
    # difine model and loss
    model = CenterNet(input_data, is_training)
    with tf.variable_scope('loss'):
        hm_loss, wh_loss, reg_loss, pose_loss = model.compute_loss(batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_ind, batch_pose)
        total_loss = hm_loss + wh_loss + reg_loss + pose_loss
    

    # define train op
    if cfg.lr_type=="CosineAnnealing":
        global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
        warmup_steps = tf.constant(cfg.warm_up_epochs * num_train_batch, dtype=tf.float64, name='warmup_steps')
        train_steps = tf.constant(cfg.epochs * num_train_batch, dtype=tf.float64, name='train_steps')
        learning_rate = tf.cond(
            pred=global_step < warmup_steps,
            true_fn=lambda: global_step / warmup_steps * cfg.init_lr,
            false_fn=lambda: cfg.end_lr + 0.5 * (cfg.init_lr - cfg.end_lr) *
                                (1 + tf.cos(
                                    (global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
        )
        global_step_update = tf.assign_add(global_step, 1.0)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([optimizer, global_step_update]):
                train_op = tf.no_op()

    else:
        global_step = tf.Variable(0, trainable=False)
        if cfg.lr_type=="exponential":
            learning_rate = tf.train.exponential_decay(cfg.lr,
                                                    global_step,
                                                    cfg.lr_decay_steps,
                                                    cfg.lr_decay_rate,
                                                    staircase=True)
        elif cfg.lr_type=="piecewise":
            learning_rate = tf.train.piecewise_constant(global_step, cfg.lr_boundaries, cfg.lr_piecewise)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step=global_step)

    saver  = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    
    
    with tf.Session() as sess:
        test_epoch_loss = 0
        with tf.name_scope('summary'):
            tf.summary.scalar("learning_rate", learning_rate)
            tf.summary.scalar("hm_loss", hm_loss)
            tf.summary.scalar("wh_loss", wh_loss)
            tf.summary.scalar("reg_loss", reg_loss)
            tf.summary.scalar("pose_loss", pose_loss)
            tf.summary.scalar("total_loss", total_loss)

            test_scalar = tf.placeholder(tf.float32)
            logdir = "./log/"
            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            write_op = tf.summary.merge_all()
            summary_writer  = tf.summary.FileWriter(logdir, graph=sess.graph)
            test_summary = tf.summary.scalar("test_loss", test_scalar)
        
        # train 
        sess.run(tf.global_variables_initializer())
        if cfg.pre_train:
            load_weights(sess,'./pretrained_weights/resnet18.npy')

        writer = tf.compat.v1.summary.FileWriter("./checkpoint", sess.graph)
        #ot_nodes = ['detector/hm/Sigmoid', "detector/wh/BiasAdd", "detector/reg/BiasAdd"]
        ot_nodes = cfg.ot_nodes
        #ot_nodes = ['detector/Conv2D_1', 'detector/Conv2D_3', 'detector/Conv2D_5']
        today = datetime.today().strftime('%Y_%m_%d')
        #saver.restore(sess, "./checkpoint/2021_05_09-centernet_test_person_loss=0.7642.ckpt-80")
        #saver.restore(sess,tf.train.latest_checkpoint('checkpoint'))
        
        for epoch in range(1, 1+cfg.epochs):
            
            pbar = tqdm(range(num_train_batch))
            train_epoch_loss, test_epoch_loss = [], []
            sess.run(trainset_init_op)
            for i in pbar:
                _, summary, train_step_loss, global_step_val = sess.run(
                    [train_op, write_op, total_loss, global_step],feed_dict={is_training:True})

                train_epoch_loss.append(train_step_loss)
                summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f" %train_step_loss)

            sess.run(testset_init_op)
            for j in range(num_test_batch):
                test_step_loss = sess.run( total_loss, feed_dict={is_training:False})
                test_epoch_loss.append(test_step_loss)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            test_summ = test_summary.eval(feed_dict={test_scalar: test_epoch_loss})
            summary_writer.add_summary(test_summ, epoch)
            ckpt_file = "./checkpoint/" + today + "-centernet_test_person_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                            %(epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            
            saver.save(sess, ckpt_file, global_step=epoch)
            # frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            #     sess,
            #     sess.graph_def,
            #     ot_nodes)
            # with open('checkpoint/centernet_graph.pb', 'wb') as f:
            #     f.write(frozen_graph_def.SerializeToString())


if __name__ == '__main__': train()