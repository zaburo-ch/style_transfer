# https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py

import tensorflow as tf
import os
import threading
import numpy as np
import ipdb

dataset_dir = '../datasets/ms_coco'
output_dir = '../datasets/ms_coco_tf'
num_total_shards = 1024
num_threads = 4


def get_example(filename):
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()
    example = tf.train.Example(features=tf.train.Features(feature={
        'jpg_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data]))
    }))
    return example


def process_in_each_thread(filenames, num_shards, num_total_shards, thread_index):
    index_range = np.linspace(0, len(filenames), num_shards + 1).astype(int)
    for s in xrange(num_shards):
        shard_index = thread_index * num_shards + s
        output_name = '{:0>5}-{:0>5}'.format(shard_index, num_total_shards)
        output_path = os.path.join(output_dir, output_name)
        with tf.python_io.TFRecordWriter(output_path) as writer:
            for file_index in xrange(index_range[s], index_range[s + 1]):
                example = get_example(filenames[file_index])
                writer.write(example.SerializeToString())
        print('Done {}th shard in {}th thread.'.format(shard_index, thread_index))


def main():
    # assume that all images are jepg.
    filenames = tf.gfile.Glob(os.path.join(dataset_dir, '*.jpg'))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    coord = tf.train.Coordinator()

    threads = []
    spacing = np.linspace(0, len(filenames), num_threads + 1).astype(int)
    num_shards_in_thread = num_total_shards / num_threads
    
    for thread_index in xrange(num_threads):
        args = (filenames[spacing[thread_index]:spacing[thread_index + 1]],
                num_shards_in_thread, num_total_shards, thread_index)
        t = threading.Thread(target=process_in_each_thread, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)


if __name__ == '__main__':
    main()
