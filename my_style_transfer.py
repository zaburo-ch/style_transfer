import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave
import ipdb


slim = tf.contrib.slim
means = [123.68, 116.779, 103.939]


def preprocessing(img):
    return np.expand_dims(img, axis=0) - means


def deprocessing(img):
    return np.clip(img[0] + means, 0, 255).astype(np.uint8)


def vgg_16(inputs, reuse=False):
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


def style_transfer(content_image,
                   style_image,
                   max_width=512,
                   content_weight=1e0,
                   content_layer='conv3_3',
                   style_layers=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3'],
                   style_weight=1e-1,
                   max_iter=1000,
                   init_image_type='content',
                   vgg_16_path='../vgg_16.ckpt'):

    # resize input images
    if content_image.shape[1] > max_width:
        new_height = int(float(max_width) / content_image.shape[1]  * content_image.shape[0])
        content_image = imresize(content_image, (new_height, max_width))
    if style_image.shape != content_image.shape:
        style_image = imresize(style_image, content_image.shape[:2])

    # preprocess images with vgg format
    content_image = preprocessing(content_image)
    style_image = preprocessing(style_image)

    # Initialise of the gradient descent
    if init_image_type == 'content':
        init_image = content_image.copy()
    elif init_image_type == 'style':
        init_image = style_image.copy()
    else:
        raise NotImplementedError

    with tf.Graph().as_default():
        input_image = tf.Variable(np.zeros_like(content_image), dtype=tf.float32, name="image")
        endpoints = vgg_16(input_image)

        with tf.Session() as sess:
            # load the trained weights of vgg-16
            saver = tf.train.Saver(slim.get_variables('vgg_16'))
            saver.restore(sess, vgg_16_path)        

            def content_loss():
                sess.run(input_image.assign(content_image))
                F = endpoints[content_layer]
                P = sess.run(F)
                return tf.reduce_mean((F - P) ** 2)

            def gram_matrix(F, N, M):
                F = tf.reshape(F, (N, M))
                return tf.matmul(F, F, transpose_a=True)

            def style_loss():
                sess.run(input_image.assign(style_image))
                loss = 0.
                layer_weight = 1. / len(style_layers)
                for style_layer in style_layers:
                    _, h, w, c = endpoints[style_layer].get_shape()
                    N = h.value * w.value
                    M = c.value
                    G = gram_matrix(endpoints[style_layer], N, M) / N
                    A = sess.run(G)
                    loss += layer_weight * tf.reduce_mean((G - A) ** 2)
                return loss

            # define the total loss tensor.
            total_loss = 0.
            total_loss += content_weight * content_loss()
            total_loss += style_weight * style_loss()

            # get L-BFGS optimizer
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                total_loss, method='L-BFGS-B',
                options={'maxiter': 1000, 'disp': 1})

            # optimize
            sess.run(input_image.assign(init_image))
            optimizer.minimize(sess)

            # get result as image
            result = deprocessing(sess.run(input_image))

    return result
