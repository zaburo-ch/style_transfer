import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import os

import loss_functions

slim = tf.contrib.slim

dataset_dir = '../datasets/ms_coco_tf'
style_image_path = '../style_input/starry-night.jpg'
vgg_ckpt_path = '../vgg_16.ckpt'


def ms_coco_inputs(batch_size, image_size, num_preprocess_threads=4):
    with tf.name_scope('batch_processing'):
        data_files = tf.gfile.Glob(os.path.join(dataset_dir, "*"))
        if data_files is None:
            raise IOError('No data files found')

        # Create filename_queue
        filename_queue = tf.train.string_input_producer(data_files,
                                                        shuffle=True,
                                                        capacity=16)

        if num_preprocess_threads % 4:
            raise ValueError('Please make num_preprocess_threads a multiple '
                             'of 4 (%d % 4 != 0).', num_preprocess_threads)

        reader = tf.TFRecordReader()
        _, example_serialized = reader.read(filename_queue)

        image_list = []
        for _ in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            image_buffer = _parse_example_proto(example_serialized)
            image = preprocess_image(image_buffer, image_size)
            image_list.append([image])

        images = tf.train.batch_join(image_list,
                                     batch_size=batch_size,
                                     capacity=2 * num_preprocess_threads * batch_size)

        images = tf.reshape(images, shape=[batch_size, image_size, image_size, 3])

        # Display the training images in the visualizer.
        tf.image_summary('images', images)

        return images


def _parse_example_proto(example_serialized):
    feature_map = {'jpg_data': tf.FixedLenFeature([], dtype=tf.string, default_value='')}
    return tf.parse_single_example(example_serialized, feature_map)['jpg_data']


def style_input(image_size, style_layers):
    with tf.gfile.FastGFile(style_image_path, 'r') as f:
        image_buffer = f.read()
    image = preprocess_image(image_buffer, image_size)
    image = tf.expand_dims(image, 0)
    
    tf.image_summary('style', image)
    
    with tf.Session() as sess:
        end_points = loss_functions.vgg_16(image)
        saver = tf.train.Saver(slim.get_variables('vgg_16'))
        saver.restore(sess, vgg_ckpt_path)
        return {key: sess.run(loss_functions.gram_matrix(end_points[key])) for key in style_layers}


def preprocess_image(image_buffer, image_size):
    image = _decode_jpeg(image_buffer)
    image = _aspect_preserving_resize(image, image_size + 2)
    image = _central_crop([image], image_size, image_size)[0]
    image.set_shape([image_size, image_size, 3])
    image = tf.to_float(image) # / 255.0
    return image


# The following functions are copied over from
# tf.slim.preprocessing.vgg_preprocessing
# because they're not visible to this module.
def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.
  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.
  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.
  Returns:
    the cropped (and resized) image.
  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    cropped_shape = control_flow_ops.with_dependencies(
        [rank_assertion],
        tf.pack([crop_height, crop_width, original_shape[2]]))

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.pack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    image = control_flow_ops.with_dependencies(
        [size_assertion],
        tf.slice(image, offsets, cropped_shape))
    return tf.reshape(image, cropped_shape)


def _central_crop(image_list, crop_height, crop_width):
    """Performs central crops of the given image list.
  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.
  Returns:
    the list of cropped images.
  """
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2

        outputs.append(_crop(image, offset_height, offset_width,
                             crop_height, crop_width))
    return outputs


def _smallest_size_at_least(height, width, smallest_side):
    """Computes new shape with the smallest side equal to `smallest_side`.
  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.
  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
    """Resize images preserving the original aspect ratio.
  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                             align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image


def _decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.
  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
    with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image