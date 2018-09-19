# DEVICE_ID = 0
# For running the model in GPU, set DEVICE_ID >= 0
DEVICE_ID = -1
import caffe

import utils
from collections import OrderedDict
import argparse
import cPickle as pickle
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import pdb

sys.path.append('../../python/')
from extract_vgg_features import *

# UNK_IDENTIFIER is the word used to identify unknown words
UNK_IDENTIFIER = 'unk_word'

word_list = list([])
  
def predict_single_word(net, mean_pool_fc7, previous_word, output='probs'):
  cont_input = 0 if previous_word==0 else 1
  cont = np.array([cont_input])
  data_en = np.array([previous_word])
  image_features = np.zeros_like(net.blobs['mean_fc7'].data)
  image_features[:] = mean_pool_fc7
  net.forward(mean_fc7=image_features, cont_sentence=cont, input_sentence=data_en)
  output_preds = net.blobs[output].data.reshape(-1)
  return output_preds

def predict_single_word_from_all_previous(net, mean_pool_fc7, previous_words):
  probs = predict_single_word(net, mean_pool_fc7, 0)
  for index, word in enumerate(previous_words):
    probs = predict_single_word(net, mean_pool_fc7, word)
  return probs

# Strategy must be either 'beam' or 'sample'.
# If 'beam', do a max likelihood beam search with beam size num_samples.
# Otherwise, sample with temperature temp.
def predict_image_caption(net, mean_pool_fc7, vocab_list, strategy={'type': 'beam'}):
  assert 'type' in strategy
  assert strategy['type'] in ('beam', 'sample')
  if strategy['type'] == 'beam':
    return predict_image_caption_beam_search(net, mean_pool_fc7, vocab_list, strategy)
  num_samples = strategy['num'] if 'num' in strategy else 1
  samples = []
  sample_probs = []
  for _ in range(num_samples):
    sample, sample_prob = sample_image_caption(net, mean_pool_fc7, strategy)
    samples.append(sample)
    sample_probs.append(sample_prob)
  return samples, sample_probs

def random_choice_from_probs(softmax_inputs, temp=1.0, already_softmaxed=False):
  if already_softmaxed:
    probs = softmax_inputs
    assert temp == 1.0
  else:
    probs = softmax(softmax_inputs, temp)
  r = random.random()
  cum_sum = 0.
  for i, p in enumerate(probs):
    cum_sum += p
    if cum_sum >= r: return i
  return 1  # return UNK?

def sample_image_caption(net, image, strategy, net_output='predict-multimodal', max_length=20):
  sentence = []
  probs = []
  eps_prob = 1e-8
  temp = strategy['temp'] if 'temp' in strategy else 1.0
  if max_length < 0: max_length = float('inf')
  while len(sentence) < max_length and (not sentence or sentence[-1] != 0):
    previous_word = sentence[-1] if sentence else 0
    softmax_inputs = \
        predict_single_word(net, image, previous_word, output=net_output)
    word = random_choice_from_probs(softmax_inputs, temp)
    sentence.append(word)
    probs.append(softmax(softmax_inputs, 1.0)[word])
  return sentence, probs

def predict_image_caption_beam_search(net, mean_pool_fc7, vocab_list, strategy, max_length=50):
  beam_size = strategy['beam_size'] if 'beam_size' in strategy else 1
  assert beam_size >= 1
  beams = [[]]
  beams_complete = 0
  beam_probs = [[]]
  beam_log_probs = [0.]
  current_input_word = 0  # first input is EOS
  while beams_complete < len(beams):
    expansions = []
    for beam_index, beam_log_prob, beam in \
        zip(range(len(beams)), beam_log_probs, beams):
      if beam:
        previous_word = beam[-1]
        if len(beam) >= max_length or previous_word == 0:
          exp = {'prefix_beam_index': beam_index, 'extension': [],
                 'prob_extension': [], 'log_prob': beam_log_prob}
          expansions.append(exp)
          # Don't expand this beam; it was already ended with an EOS,
          # or is the max length.
          continue
      else:
        previous_word = 0  # EOS is first word
      if beam_size == 1:
        probs = predict_single_word(net, mean_pool_fc7, previous_word)
      else:
        probs = predict_single_word_from_all_previous(net, mean_pool_fc7, beam)
      assert len(probs.shape) == 1
      assert probs.shape[0] == len(vocab_list)
      expansion_inds = probs.argsort()[-beam_size:]
      for ind in expansion_inds:
        prob = probs[ind]
        extended_beam_log_prob = beam_log_prob + math.log(prob)
        exp = {'prefix_beam_index': beam_index, 'extension': [ind],
               'prob_extension': [prob], 'log_prob': extended_beam_log_prob}
        expansions.append(exp)
    # Sort expansions in decreasing order of probabilitf.
    expansions.sort(key=lambda expansion: -1 * expansion['log_prob'])
    expansions = expansions[:beam_size]
    new_beams = \
        [beams[e['prefix_beam_index']] + e['extension'] for e in expansions]
    new_beam_probs = \
        [beam_probs[e['prefix_beam_index']] + e['prob_extension'] for e in expansions]
    beam_log_probs = [e['log_prob'] for e in expansions]
    beams_complete = 0
    for beam in new_beams:
      if beam[-1] == 0 or len(beam) >= max_length: beams_complete += 1
    beams, beam_probs = new_beams, new_beam_probs
  return beams, beam_probs

def run_pred_iter(net, mean_pool_fc7, vocab_list, strategies=[{'type': 'beam'}]):
  outputs = []
  for strategy in strategies:
    captions, probs = predict_image_caption(net, mean_pool_fc7, vocab_list, strategy=strategy)
    for caption, prob in zip(captions, probs):
      output = {}
      output['caption'] = caption
      output['prob'] = prob
      output['gt'] = False
      output['source'] = strategy
      outputs.append(output)
  return outputs

def score_caption(net, image, caption, is_gt=True, caption_source='gt'):
  output = {}
  output['caption'] = caption
  output['gt'] = is_gt
  output['source'] = caption_source
  output['prob'] = []
  probs = predict_single_word(net, image, 0)
  for word in caption:
    output['prob'].append(probs[word])
    probs = predict_single_word(net, image, word)
  return output

def gen_stats(prob, normalizer=None):
  stats = {}
  stats['length'] = len(prob)
  stats['log_p'] = 0.0
  eps = 1e-12
  for p in prob:
    assert 0.0 <= p <= 1.0
    stats['log_p'] += math.log(max(eps, p))
  stats['log_p_word'] = stats['log_p'] / stats['length']
  try:
    stats['perplex'] = math.exp(-stats['log_p'])
  except OverflowError:
    stats['perplex'] = float('inf')
  try:
    stats['perplex_word'] = math.exp(-stats['log_p_word'])
  except OverflowError:
    stats['perplex_word'] = float('inf')
  if normalizer is not None:
    norm_stats = gen_stats(normalizer)
    stats['normed_perplex'] = \
        stats['perplex'] / norm_stats['perplex']
    stats['normed_perplex_word'] = \
        stats['perplex_word'] / norm_stats['perplex_word']
  return stats

def run_pred_iters(pred_net, image_list, feature_extractor,
                   strategies=[{'type': 'beam'}], display_vocab=None):
  outputs = OrderedDict()
  num_pairs = 0
  descriptor_video_id = ''
  # print "CNN ..."
  features = feature_extractor.compute_features(image_list)
  for index, video_id in enumerate(image_list):
    assert video_id not in outputs
    num_pairs += 1
    if descriptor_video_id != video_id:
      image_fc7 = features[index]
      desciptor_video_id = video_id
    outputs[video_id] = \
        run_pred_iter(pred_net, image_fc7, display_vocab, strategies=strategies)
    if display_vocab is not None:
      for output in outputs[video_id]:
        caption, prob, gt, source = \
            output['caption'], output['prob'], output['gt'], output['source']
        caption_string = utils.vocab_inds_to_sentence(display_vocab, caption)
        if gt:
          tag = 'Actual'
        else:
          tag = 'Generated'
        if not 'stats' in output:
          stats = gen_stats(prob)
          output['stats'] = stats
        stats = output['stats']
        print '%s caption (length %d, log_p = %f, log_p_word = %f):\n%s' % \
            (tag, stats['length'], stats['log_p'], stats['log_p_word'], caption_string)
  return outputs


def compute_descriptors(net, image_list, output_name='fc7'):
  batch = np.zeros_like(net.blobs['data'].data)
  batch_shape = batch.shape
  batch_size = batch_shape[0]
  descriptors_shape = (len(image_list), ) + net.blobs[output_name].data.shape[1:]
  descriptors = np.zeros(descriptors_shape)
  for batch_start_index in range(0, len(image_list), batch_size):
    batch_list = image_list[batch_start_index:(batch_start_index + batch_size)]
    for batch_index, image_path in enumerate(batch_list):
      batch[batch_index:(batch_index + 1)] = preprocess_image(net, image_path)
    print 'Computing descriptors for images %d-%d of %d' % \
        (batch_start_index, batch_start_index + batch_size - 1, len(image_list))
    net.forward(data=batch)
    # print 'Done'
    descriptors[batch_start_index:(batch_start_index + batch_size)] = \
        net.blobs[output_name].data
  return descriptors

def softmax(softmax_inputs, temp):
  exp_inputs = np.exp(temp * softmax_inputs)
  exp_inputs_sum = exp_inputs.sum()
  if math.isnan(exp_inputs_sum):
    return exp_inputs * float('nan')
  elif math.isinf(exp_inputs_sum):
    assert exp_inputs_sum > 0  # should not be -inf
    return np.zeros_like(exp_inputs)
  eps_sum = 1e-8
  return exp_inputs / max(exp_inputs_sum, eps_sum)

def sample_captions(net, image_features,
    prob_output_name='probs', output_name='samples', caption_source='sample'):
  cont_input = np.zeros_like(net.blobs['cont_sentence'].data)
  word_input = np.zeros_like(net.blobs['input_sentence'].data)
  batch_size = image_features.shape[0]
  outputs = []
  output_captions = [[] for b in range(batch_size)]
  output_probs = [[] for b in range(batch_size)]
  caption_index = 0
  num_done = 0
  while num_done < batch_size:
    if caption_index == 0:
      cont_input[:] = 0
    elif caption_index == 1:
      cont_input[:] = 1
    if caption_index == 0:
      word_input[:] = 0
    else:
      for index in range(batch_size):
        word_input[index] = \
            output_captions[index][caption_index - 1] if \
            caption_index <= len(output_captions[index]) else 0
    net.forward(image_features=image_features,
        cont_sentence=cont_input, input_sentence=word_input)
    net_output_samples = net.blobs[output_name].data
    net_output_probs = net.blobs[prob_output_name].data
    for index in range(batch_size):
      # If the caption is empty, or non-empty but the last word isn't EOS,
      # predict another word.
      if not output_captions[index] or output_captions[index][-1] != 0:
        next_word_sample = net_output_samples[index]
        assert next_word_sample == int(next_word_sample)
        next_word_sample = int(next_word_sample)
        output_captions[index].append(next_word_sample)
        output_probs[index].append(net_output_probs[index, next_word_sample])
        if next_word_sample == 0: num_done += 1
    print '%d/%d done after word %d' % (num_done, batch_size, caption_index)
    caption_index += 1
    
  for prob, caption in zip(output_probs, output_captions):
    output = {}
    output['caption'] = caption
    output['prob'] = prob
    output['gt'] = False
    output['source'] = caption_source
    outputs.append(output)
  return outputs

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--modelname", type=str,
                      default="models/imgnetcoco_3loss_voc72klabel_inglove_prelm75k_sgd_lr4e5_iter_80000.caffemodel.h5",
                      help='Path to NOC model (Imagenet/CoCo).')
  parser.add_argument("-v", "--vggmodel", type=str,
                      default="models/VGG_ILSVRC_16_layers.caffemodel",
                      help='Path to vgg 16 model file.')
  parser.add_argument("-i", "--imagelist", type=str,
                      default="images_list.txt",
                      help='File with a list of images (full path to images).')
  parser.add_argument("-o", "--htmlout", action='store_true', help='output images and captions as html')
  args = parser.parse_args()

  VOCAB_FILE = './surf_intersect_glove.txt'
  LSTM_NET_FILE = './deploy.3loss_coco_fc7_voc72klabel.shared_glove.prototxt'
  VGG_NET_FILE = 'vgg_orig_16layer.deploy.prototxt'
  RESULTS_DIR = './results'
  MODEL_FILE = args.modelname
  NET_TAG = os.path.basename(args.modelname)

  if DEVICE_ID >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(DEVICE_ID)
  else:
    caffe.set_mode_cpu()
  # print "Setting up CNN..."
  feature_extractor = FeatureExtractor(args.vggmodel, VGG_NET_FILE, DEVICE_ID)
  # print "Setting up LSTM NET"
  lstm_net = caffe.Net(LSTM_NET_FILE, MODEL_FILE, caffe.TEST)
  if MODEL_FILE.endswith('.h5'):
    utils.load_weights_from_h5(lstm_net, MODEL_FILE)
  nets = [lstm_net]

  STRATEGIES = [
    {'type': 'beam', 'beam_size': 1},
#    {'type': 'sample', 'temp': 2, 'num': 25},  # CoCo held-out reported in paper.
  ]
  NUM_OUT_PER_CHUNK = 30
  START_CHUNK = 0

  vocabulary, vocabulary_inverted = utils.init_vocab_from_file(VOCAB_FILE)
  image_list = []
  assert os.path.exists(args.imagelist)
  with open(args.imagelist, 'r') as infd:
    image_list = infd.read().splitlines()
    # In this case, batch of images can be processed. Store them into a list of files.

  print 'Captioning %d images...' % len(image_list)
  NUM_CHUNKS = (len(image_list)/NUM_OUT_PER_CHUNK) + 1 # num videos in batches of 30
  eos_string = '<EOS>'
  # add english inverted vocab 
  vocab_list = [eos_string] + vocabulary_inverted
  offset = 0
  data_split_name = 'output'
  for c in range(START_CHUNK, NUM_CHUNKS):
    chunk_start = c * NUM_OUT_PER_CHUNK
    chunk_end = (c + 1) * NUM_OUT_PER_CHUNK
    chunk = image_list[chunk_start:chunk_end]
    html_out_filename = '%s/%s.%s.%d_to_%d.html' % \
        (RESULTS_DIR, data_split_name, NET_TAG, chunk_start, chunk_end)
    text_out_filename = '%s/%s.%s_' % \
        (RESULTS_DIR, data_split_name, NET_TAG)
    outputs = run_pred_iters(lstm_net, chunk, feature_extractor,
                             strategies=STRATEGIES, display_vocab=vocab_list)
    for strat_type in text_out_types:
      text_out_fname = text_out_filename + strat_type + '.txt'
      text_out_file = open(text_out_fname, 'a')
      text_out_file.write(''.join(text_out_types[strat_type]))
      text_out_file.close()
    offset += NUM_OUT_PER_CHUNK
    
    # uncomment when first excuting
    # utils.writeWordIntoFile(word_list,'~/dictionary.txt')

if __name__ == "__main__":
 main()

