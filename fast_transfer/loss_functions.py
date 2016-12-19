import tensorflow as tf
import numpy as np

import image_utils
import model

slim = tf.contrib.slim
use_magenta = False

def vgg_16(inputs, reuse=False):
    inputs *= 255.0
    inputs -= tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
    with tf.variable_scope('vgg_16', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d], trainable=False,
                            padding='SAME', activation_fn=tf.nn.relu):
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.avg_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.avg_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.avg_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.avg_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.avg_pool2d(net, [2, 2], scope='pool5')

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    return {key.split("/")[-1]:item for key, item in end_points.iteritems()}


def content_loss(stylized_end_points, content_end_points, content_weights):
    loss = np.float32(0.0)
    for name, weight in content_weights.iteritems():
        F = stylized_end_points[name]
        P = content_end_points[name]
        loss += weight * tf.reduce_mean((F - P) ** 2)
    return loss


def gram_matrix(F):
    b, h, w, c = F.get_shape()
    N = h.value * w.value
    M = c.value
    F = tf.reshape(F, (b.value, N, M))
    # transpose_a
    return tf.batch_matmul(tf.transpose(F, (0, 2, 1)), F) / tf.to_float(N)


def style_loss(stylized_end_points, style_matrices, style_weights):
    loss = np.float32(0.0)
    for name, weight in style_weights.iteritems():
        G = gram_matrix(stylized_end_points[name])
        # sm = tf.reduce_mean(G ** 2)
        # sm = tf.Print(sm, [sm], message="Gram matrix sm:")
        # tf.scalar_summary("Gram matrix sm " + name, sm)
        # m = tf.reduce_max(G)
        # m = tf.Print(m, [m], message="Gram matrix max:")
        # tf.scalar_summary("Gram matrix max " + name, m)
        A = tf.to_float(style_matrices[name])
        loss += weight * tf.reduce_mean((G - A) ** 2)
    return loss


def total_variation_loss(x, tv_weight):
    # beta = 2
    tv_x = tf.reduce_mean((x[:, :, 1:, :] - x[:, :, :-1, :]) ** 2)
    tv_y = tf.reduce_mean((x[:, 1:, :, :] - x[:, :-1, :, :]) ** 2)
    return tv_weight * (tv_x + tv_y)
    

def total_loss(inputs, style_matrices, stylized_inputs,
                content_weights, style_weights, tv_weight):
    # sm = tf.reduce_mean(stylized_inputs ** 2)
    # sm = tf.Print(sm, [sm], message="stylized_inputs sm:")
    # tf.scalar_summary("style sm", sm)

    content_end_points = vgg_16(inputs, reuse=False)
    stylized_end_points = vgg_16(stylized_inputs, reuse=True)
    
    loss_dict = {
        'content_loss': content_loss(stylized_end_points, content_end_points, content_weights),
        'style_loss': style_loss(stylized_end_points, style_matrices, style_weights),
        'tv_loss': total_variation_loss(stylized_inputs, tv_weight)}
    # loss_dict['content_loss'] = tf.Print(loss_dict['content_loss'], [loss_dict['content_loss']], message='content_loss')
    # loss_dict['style_loss'] = tf.Print(loss_dict['style_loss'], [loss_dict['style_loss']], message='style_loss')

    loss_dict['total_loss'] = loss_dict['content_loss'] + loss_dict['style_loss'] + loss_dict['tv_loss']
    return loss_dict

