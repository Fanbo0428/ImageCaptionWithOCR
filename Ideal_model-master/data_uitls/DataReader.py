# -*- coding = utf-8 -*-

from __future__ import absolute_import,division,print_function

import numpy as np
from os.path import join
import tensorflow as tf
import convert_to_tfrecords

# Path to TFrecord files (Training and validation)
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'

# Basic information of images
IMG_HEIGHT = convert_to_tfrecords.IMG_HEIGHT
IMG_WIDTH = convert_to_tfrecords.IMG_WIDTH
IMG_CHANNELS = convert_to_tfrecords.IMG_CHANNELS
IMG_PIXELS = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS

# The amount of traning and validation dataset
NUM_TRAIN = convert_to_tfrecords.NUM_TRAIN
NUM_VALIDARION = convert_to_tfrecords.NUM_VALIDARION

def read_and_decode(filename_queue):
    # Define a reader f=to read the cases in TFRecords
    reader = tf.TFRecordReader()
    # Read a case from file
    _,serialized_example = reader.read(filename_queue)
    # Analyze a case readed
    features = tf.parse_single_example(serialized_example,features={
        'label':tf.FixedLenFeature([],tf.int64),
        'image_raw':tf.FixedLenFeature([],tf.string)
        })
    # Convert images string back to pixels
    image = tf.decode_raw(features['image_raw'],tf.uint8)
    label = tf.cast(features['label'],tf.int32)

    image.set_shape([IMG_PIXELS])
    image = tf.reshape(image,[IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    return image,label


def inputs(data_set,batch_size,num_epochs):
    if not num_epochs:
        num_epochs = None
    if data_set == 'train':
        file = TRAIN_FILE
    else:
        file = VALIDATION_FILE

    with tf.name_scope('input') as scope:
        filename_queue = tf.train.string_input_producer([file], num_epochs=num_epochs)
    image,label = read_and_decode(filename_queue)
    images,labels = tf.train.shuffle_batch([image, label], 
        batch_size=batch_size,
        num_threads=64,
        capacity=1000 + 3 * batch_size,
        min_after_dequeue=1000
    )
    return images,labels