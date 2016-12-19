import tensorflow as tf
import numpy as np

import image_utils
import loss_functions
import os
import sys
import datetime

from magenta.models.image_stylization import model

slim = tf.contrib.slim

ps_task = 0
# content_weights = {'conv4_2': 1e-5}
# style_weights = {name: 1e-8 for name in ['conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']}
content_weights = {'conv3_3': 1.0}
style_weights = {name: 2e-4 for name in ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']}
tv_weight = 1e5

batch_size = 4
image_size = 256
learning_rate = 1e-3
train_steps = 40000

logdir = "run_" + datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
vgg_ckpt_path = '../vgg_16.ckpt'


with tf.Graph().as_default():
    # Force all input processing onto CPU in order to reserve the GPU for the
    # forward inference and back-propagation.
    device = '/cpu:0'
    with tf.device(tf.train.replica_device_setter(ps_task, worker_device=device)):
        inputs = image_utils.ms_coco_inputs(batch_size, image_size)

    with tf.device(tf.train.replica_device_setter(ps_task)):
        # style_matrices = image_utils.style_input(image_size, style_weights.keys())
        # np.savez("style_matrices_convn_1.npz", **style_matrices)
        # sys.exit(0)
        # style_matrices = np.load('style_matrices_convn_1.npz')
        style_matrices = np.load('style_matrices.npz')

        stylized_inputs = model.transform(
            inputs, normalizer_fn=slim.batch_norm,
            normalizer_params={})
        tf.image_summary('stylized', stylized_inputs)

        loss_dict = loss_functions.total_loss(
            inputs, style_matrices, stylized_inputs,
            content_weights, style_weights, tv_weight)
        for key, value in loss_dict.iteritems():
            tf.scalar_summary(key, value)

        """
        # model test loss
        model_test_loss = tf.reduce_mean((inputs - stylized_inputs) ** 2)
        tf.scalar_summary('model_test_loss', model_test_loss)
        stylized_sm = tf.reduce_mean(stylized_inputs ** 2)
        tf.scalar_summary('stylized_sm', stylized_sm)
        input_sm = tf.reduce_mean(inputs ** 2)
        tf.scalar_summary('input_sm', input_sm)
        """
        
        # Set up training
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = slim.learning.create_train_op(loss_dict['total_loss'], optimizer)
        # train_op = slim.learning.create_train_op(model_test_loss, optimizer)

        init_fn = slim.assign_from_checkpoint_fn(vgg_ckpt_path, slim.get_model_variables('vgg_16'))

        # Run training
        slim.learning.train(
            train_op=train_op,
            logdir=os.path.expanduser(logdir),
            number_of_steps=train_steps,
            init_fn=init_fn,
            save_summaries_secs=10,
            save_interval_secs=10)
