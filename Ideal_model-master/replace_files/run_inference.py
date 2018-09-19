
# -*- coding: utf-8 -*-
#This part of code is modified from the orignial version of Goolge's NIC model. Some functions and improvements are added for model assembling usage
#See the original license in
#http://www.apache.org/licenses/LICENSE-2.0
#####
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary


import math
import os
import sys
import getopt

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_files", "/Users/apple/Desktop/test7.jpeg",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
# Here set a default image path for test use, in running model, the inputfiles are defiend by -i tags
tf.flags.DEFINE_string("checkpoint_path", "/Users/apple/Desktop/IndividualProject/image_caption/runned_model/Show_And_Tell/data/mscoco/train",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "/Users/apple/Desktop/IndividualProject/image_caption/runned_model/Show_And_Tell/data/mscoco/raw-data/word_counts.txt", "Text file containing the vocabulary.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(argv):
    inputfile = ' '
    outputfile = ' '
    try:
         opts, args = getopt.getopt(argv,"hi:o",["ifile=","ofile="])
    except getopt.GetoptError:
         print("input/output error ")
         sys.exit(2)
    for opt, arg in opts:
         if opt =='-h':
              print ('usage: python run_inference.py -i <inputfile> -o <outuptfile>')
              sys.exit()
         elif opt in ('-i','--input'):
              inputfile = arg
         elif opt in ('-o','--output'):
              outputfile = arg 
    
    # print(inputfile)
    
    g = tf.Graph()
    # Build the inference graph.
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                   FLAGS.checkpoint_path)
    g.finalize()

    # Create the vocabulary.
    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

    filenames = []
    if inputfile == ' ':
         for file_pattern in FLAGS.input_files.split(","):
             filenames.extend(tf.gfile.Glob(file_pattern))
    else:
         for file_pattern in inputfile.split(","):
             filenames.extend(tf.gfile.Glob(file_pattern))
    # print (filenames)
    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)
        
        generator = caption_generator.CaptionGenerator(model, vocab)

        for filename in filenames:
            with tf.gfile.FastGFile(filename, "rb") as f:
                image = f.read()
            captions = generator.beam_search(sess, image)
            # print("Captions for image %s :" % os.path.basename(filename))
            # print(captions)
            prob = []
            for i, caption in enumerate(captions):
                # Ignore begin and end words.
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                prob.append(caption.logprob)
                # print (sentence)
            # In this case, only the one with the largetst logprob is left for futher operation
            for caption in captions:
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                if 'UNK' in sentence:# if luckily the model recognized the text information itself
                    final = sentence 
                    break
                if caption.logprob == max(prob):
                    final = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                    final = ' '.join(final)
            img = Image.open(FLAGS.input_files)
            print("%s : " % os.path.basename(filename) + final)
            # print(final)
            # plt.imshow(img)
            # plt.axis('off')
            # plt.title(str(final))
            # plt.show()


if __name__ == "__main__":
    tf.app.run()
