# -*- coding: utf-8 -*-
#This part of code is modified from the orignial version of TextBoxes. Some functions and improvements are added for model assembling usage
# The original paper and code in bibtex format is below
    # @inproceedings{LiaoSBWL17,
    #   author    = {Minghui Liao and
    #                Baoguang Shi and
    #                Xiang Bai and
    #                Xinggang Wang and
    #                Wenyu Liu},
    #   title     = {TextBoxes: {A} Fast Text Detector with a Single Deep Neural Network},
    #   booktitle = {AAAI},
    #   year      = {2017}
    # }
# For running codes in this project, this file should replace the orinignal file named 'demo.py' in directory 'exmples'

import os
import sys
import getopt
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xml.dom.minidom
from nms import nms
from PIL import Image

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#caffe_root = '/path/to/TextBoxes/'
caffe_root = '/Users/apple/Desktop/TextBoxes/'

os.chdir(caffe_root)
sys.path.append(0,'python')

#Here caffe is imported after the running directory is located into TextBoxes
import caffe

model_def = '~/deploy.prototxt'
model_weights = '~/TextBoxes_icdar13.caffemodel'

use_multi_scale = True

def main(argv):
         inputfile= ' '
         outputfile=' '
         try:
              opts, args = getopt.getopt(argv,"hi:o",["ifile=","ofile="])
         except getopt.GetoptError:
              print 'input/output error'
              sys.exit(2)
         for opt, arg in opts:
              if opt=='-h':
                   # print 'help'
                   ## instructions goes here
                   sys.exit()
              elif opt in ('-i',"--ifile"):
                   inputfile = arg
              elif opt in ("-o","--ofile"):
                   outputfile = arg
         logging.info('input file: %s', inputfile)
         
    if not use_multi_scale:
        scales=((700,700),)
    else:
        scales=((300,300),(700,700),(700,500),(700,300),(1600,1600))
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    dt_results=[]
    image_path=inputfile
    image=caffe.io.load_image(image_path)
    image_height,image_width,channels=image.shape
    plt.clf()
    plt.imshow(image)
    currentAxis = plt.gca()
    for scale in scales:
        image_resize_height = scale[0]
        image_resize_width = scale[1]
        transformer = caffe.io.Transformer({'data': (1,3,image_resize_height,image_resize_width)})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104,117,123])) # mean pixel
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

        net.blobs['data'].reshape(1,3,image_resize_height,image_resize_width)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        # Forward pass.
        detections = net.forward()['detection_out']
        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
        top_conf = det_conf[top_indices]
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        for i in xrange(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            xmin = max(1,xmin)
            ymin = max(1,ymin)
            xmax = min(image.shape[1]-1, xmax)
            ymax = min(image.shape[0]-1, ymax)
            # score = top_conf[i]
            dt_result=[xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]
            # comment the score part for simplicity on fetching the text in image.
            dt_results.append(dt_result)

    # dt_results = sorted(dt_results, key=lambda x:-float(x[8]))
    nms_flag = nms(dt_results,0.3)

    for k,dt in enumerate(dt_results):
          if nms_flag[k]:
            # name = '%.2f'%(dt[8])
            xmin = dt[0]
            ymin = dt[1]
            xmax = dt[2]
            ymax = dt[5]
            #The co-ordinates containing the pixels with text.
            # coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            ####cut the text in the image########
            x = xmin
            y = ymin
            w = xmax-xmin+1
            h = ymax-ymin+1
            im = Image.open(inputfile)
            text_region = im.crop((x, y, x+w, y+h))
            text_region.save(outputfile)
            ########cut the text in the image####
if __name__ =="__main__":
     main(sys.argv[1:])
