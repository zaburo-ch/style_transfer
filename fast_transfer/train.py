import tensorflow as tf
import numpy as np

import image_utils
import loss_functions
import os
import sys
import datetime

# TODO: Define the model yourself
from magenta.models.image_stylization import model

slim = tf.contrib.slim

# TODO: Use argparse
ps_task = 0
content_weights = {'conv3_3': 1.0}
style_weights = {name: 2e-4 for name in ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']}
tv_weight = 0

batch_size = 4
image_size = 256
learning_rate = 1e-3
train_steps = 40000

logdir = "run_" + datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
vgg_ckpt_path = '../vgg_16.ckpt'


with tf.Graph().as_default():
    device = '/cpu:0'
    with tf.device(tf.train.replica_device_setter(ps_task, worker_device=device)):
        inputs = image_utils.ms_coco_inputs(batch_size, image_size)

    with tf.device(tf.train.replica_device_setter(ps_task)):
        if os.path.exists('style_matrices.npz'):
            style_matrices = image_utils.style_input(image_size, style_weights.keys())
            np.savez('style_matrices.npz', **style_matrices)
        else:
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

        # Set up training
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = slim.learning.create_train_op(loss_dict['total_loss'], optimizer)

        init_fn = slim.assign_from_checkpoint_fn(vgg_ckpt_path, slim.get_model_variables('vgg_16'))

        # Run training
        slim.learning.train(
            train_op=train_op,
            logdir=os.path.expanduser(logdir),
            number_of_steps=train_steps,
            init_fn=init_fn,
            save_summaries_secs=10,
            save_interval_secs=10)
