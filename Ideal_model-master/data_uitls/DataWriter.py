# -*- coding: utf-8 -*-

from __future__ import absolute_import,division,print_function

import numpy as np
import tensorflow as tf
import time
from scipy.misc import imread,imresize
from os import  walk
from os.path import join

#directory for stroing the traning data
DATA_DIR = '/Users/apple/Desktop/img/'

#image information, note that it must be the same with NIC requirements

IMG_HEIGHT = 227
IMG_WIDTH = 227
IMG_CHANNELS = 3
NUM_TRAIN = 376
NUM_VALIDARION = 0

# For int attribute
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# For string attribute
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Function for read images
def read_images(path):
    filenames = next(walk(path))[2]
    num_files = len(filenames)
    images = np.zeros((num_files,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
    labels = list()
    f = open('sorted_template_captions.txt')
    lines = f.readlines()
    #遍历所有的图片和label，将图片resize到[227,227,3]
    for i,filename in enumerate(filenames):
        img = imread(join(path,filename))
        # print ('Image ' + filename +' Readed!')
        img = imresize(img,(IMG_HEIGHT,IMG_WIDTH))
        images[i] = img
        index = lines[i].index(':')
        labels.append(str(lines[i][index+2:]))
        # labels[i] = lines[i]
    f.close()
    return images,labels
# print(read_images(DATA_DIR)[1])

def convert(images,labels,name):
    # Get the numer of images for converting to TFrecord
    num = images.shape[0]
    # Filenames for Tfrecord
    filename = name+'.tfrecords'
    print('Writting',filename)
    # Define a writer for wriring TFRecord files
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(num):
        # Convert matrix of image into string
        img_raw = images[i].tostring()
        # Turn a case into Example Protocol Buffer，then wirite all the information into data structure
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _bytes_feature(labels[i]),
            'image_raw': _bytes_feature(img_raw)}))
        # Wirte example into TFRecord files
        writer.write(example.SerializeToString())
    writer.close()
    print('Writting End')
    

# Main operations goes there
def main(argv):
    print('reading images begin')
    start_time = time.time()
    train_images,train_labels = read_images(DATA_DIR)
    duration = time.time() - start_time
    print("reading images end , cost %d sec" %duration)

    #get validation
    validation_images = train_images[:NUM_VALIDARION,:,:,:]
    validation_labels = train_labels[:NUM_VALIDARION]
    train_images = train_images[NUM_VALIDARION:,:,:,:]
    train_labels = train_labels[NUM_VALIDARION:]

    #convert to tfrecords
    print('convert to tfrecords begin')
    start_time = time.time()
    convert(train_images,train_labels,'train')
    convert(validation_images,validation_labels,'validation')
    duration = time.time() - start_time
    print('convert to tfrecords end , cost %d sec' %duration)

if __name__ == '__main__':
    tf.app.run()