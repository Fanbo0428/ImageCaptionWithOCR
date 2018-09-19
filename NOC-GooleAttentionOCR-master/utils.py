# !/usr/bin/python
# -*- coding: UTF-8 -*-
import os 
import system


def  writeWordIntoFile(word_list,filename): 
    with open(filename,'w') as f:
        for w in word_list:
            f.write(w)
            
def read_caption(textFile):
  with open(textFile) as f:
    data = f.readlines()
  return data

def init_vocab_from_file(vocab_filename):
  # print "Initializing vocabulary from %s ..." % vocab_filename
  vocabulary = {}
  vocabulary_inverted = []
  if os.path.isfile(vocab_filename):
    with open(vocab_filename, 'rb') as vocab_filedes:
      # initialize the vocabulary with the UNK_IDENTIFIERK word
      vocabulary = {UNK_IDENTIFIER: 0}
      vocabulary_inverted = [UNK_IDENTIFIER]
      num_words_dataset = 0
      for line in vocab_filedes.readlines():
        split_line = line.split()
        word = split_line[0]
        word_list.append(word)
        if word == UNK_IDENTIFIER:
          continue
        else:
          assert word not in vocabulary
        num_words_dataset += 1
        vocabulary[word] = len(vocabulary_inverted)
        vocabulary_inverted.append(word)
      num_words_vocab = len(vocabulary.keys())
      # print ('Initialized vocabulary from file with %d unique words ' +
      #        '(from %d total words in dataset).') % \
      #       (num_words_vocab, num_words_dataset)
      assert len(vocabulary_inverted) == num_words_vocab
  else:
    print('Vocabulary file %s does not exist' % vocab_filename)
  return vocabulary, vocabulary_inverted

def vocab_inds_to_sentence(vocab, inds):
  sentence = ' '.join([vocab[i] for i in inds])
  # Capitalize first character.
  sentence = sentence[0].upper() + sentence[1:]
  # Replace <EOS> with '.', or append '...'.
  if sentence.endswith(' ' + vocab[0]):
    sentence = sentence[:-(len(vocab[0]) + 1)] + '.'
  else:
    sentence += '...'
  return sentence
            
def load_weights_from_h5(net_object, h5_weights_file):
  h5_weights = h5py.File(h5_weights_file)
  for layer in net_object.params.keys():
    assert layer in h5_weights['data'].keys()
    num_axes = np.shape(net_object.params[layer])[0]
    wgt_axes = h5_weights['data'][layer].keys()
    assert num_axes == len(wgt_axes)
    for axis in range(num_axes):
      net_object.params[layer][axis].data[:] = h5_weights['data'][layer][wgt_axes[axis]]
      

def to_html_row(columns, header=False):
  out= '<tr>'
  for column in columns:
    if header: out += '<th>'
    else: out += '<td>'
    try:
      if int(column) < 1e8 and int(column) == float(column):
        out += '%d' % column
      else:
        out += '%0.04f' % column
    except:
      out += '%s' % column
    if header: out += '</th>'
    else: out += '</td>'
  out += '</tr>'
  return out
#In the original model, the output can be shown in web page. Thus the HTML is generated. However in this case, this function is uneeded, so the function is archived in here.
def to_html_output(outputs, vocab):
  out = ''
  for video_id, captions in outputs.iteritems():
    for c in captions:
      if not 'stats' in c:
        c['stats'] = gen_stats(c['prob'])
    # Sort captions by log probability.
    if 'normed_perplex' in captions[0]['stats']:
      captions.sort(key=lambda c: c['stats']['normed_perplex'])
    else:
      captions.sort(key=lambda c: -c['stats']['log_p_word'])
    out += '<img src="%s"><br>\n' % video_id
    out += '<table border="1">\n'
    column_names = ('Source', '#Words', 'Perplexity/Word', 'Caption')
    out += '%s\n' % to_html_row(column_names, header=True)
    for c in captions:
      caption, gt, source, stats = \
          c['caption'], c['gt'], c['source'], c['stats']
      caption_string = vocab_inds_to_sentence(vocab, caption)
      if gt:
        source = 'ground truth'
        if 'correct' in c:
          caption_string = '<font color="%s">%s</font>' % \
              ('green' if c['correct'] else 'red', caption_string)
        else:
          caption_string = '<em>%s</em>' % caption_string
      else:
        if source['type'] == 'beam':
          source = 'beam (size %d)' % source['beam_size']
        elif source['type'] == 'sample':
          source = 'sample (temp %f)' % source['temp']
        else:
          raise Exception('Unknown type: %s' % source['type'])
        caption_string = '<strong>%s</strong>' % caption_string
      columns = (source, stats['length'] - 1,
                 stats['perplex_word'], caption_string)
      out += '%s\n' % to_html_row(columns)
    out += '</table>\n'
    out += '<br>\n\n' 
    out += '<br>' * 2
  out.replace('<unk>', 'UNK')  
  return out


def to_text_output(outputs, vocab):
  out_types = {}
  caps = outputs[outputs.keys()[0]]
  for c in caps:
    caption, gt, source = \
        c['caption'], c['gt'], c['source']
    if source['type'] == 'beam':
      source_meta = 'beam_size_%d' % source['beam_size']
    elif source['type'] == 'sample':
      source_meta = 'sample_temp_%.2f' % source['temp']
    else:
      raise Exception('Unknown type: %s' % source['type'])
    if source_meta not in out_types:
      out_types[source_meta] = []
  num_videos = 0
  out = ''
  for video_id, captions in outputs.iteritems():
    num_videos += 1
    for c in captions:
      if not 'stats' in c:
        c['stats'] = gen_stats(c['prob'])
    # Sort captions by log probability.
    if 'normed_perplex' in captions[0]['stats']:
      captions.sort(key=lambda c: c['stats']['normed_perplex'])
    else:
      captions.sort(key=lambda c: -c['stats']['log_p_word'])
    for c in captions:
      caption, gt, source, stats = \
          c['caption'], c['gt'], c['source'], c['stats']
      caption_string = vocab_inds_to_sentence(vocab, caption)
      if source['type'] == 'beam':
        source_meta = 'beam_size_%d' % source['beam_size']
      elif source['type'] == 'sample':
        source_meta = 'sample_temp_%.2f' % source['temp']
      else:
        raise Exception('Unknown type: %s' % source['type'])
      out = '%s\t%s\tlog_p=%f, log_p_word=%f\t%s\n' % (source_meta, video_id,
        c['stats']['log_p'], c['stats']['log_p_word'], caption_string)
      out_types[source_meta].append(out)
  return out_types
  
  
  
## For video caption, the funtions are archived below

def next_video_gt_pair(tsg):
  # modify to return a list of frames and a stream for the hdf5 outputs
  streams = tsg.get_streams()
  video_id = tsg.lines[tsg.line_index-1][0]
  gt = streams['target_sentence']
  return video_id, gt

# keep all frames for the video (including padding frame)
def all_video_gt_pairs(fsg):
  data = OrderedDict()
  if len(fsg.lines) > 0:
    prev_video_id = None
    while True:
      video_id, gt = next_video_gt_pair(fsg)
      if video_id in data:
        if video_id != prev_video_id:
          break
        data[video_id].append(gt)
      else:
        data[video_id] = [gt]
      prev_video_id = video_id
    print 'Found %d videos with %d captions' % (len(data.keys()), len(data.values()))
  else:
    data = OrderedDict(((key, []) for key in fsg.vid_poolfeats.keys()))
  return data
  
def video_to_descriptor(video_id, fsg):
  video_features = []
  assert video_id in fsg.vid_poolfeats
  text_mean_fc7 = fsg.vid_poolfeats[video_id][0]
  mean_fc7 = fsg.float_line_to_stream(text_mean_fc7)
  pool_feature = np.array(mean_fc7).reshape(1, 1, len(mean_fc7))
  return pool_feature
 
## For video caption, the funtions are archived above




def get_crop_size():
  if FLAGS.crop_width and FLAGS.crop_height:
    return (FLAGS.crop_width, FLAGS.crop_height)
  else:
    return None


def create_dataset(split_name):
  ds_module = getattr(datasets, FLAGS.dataset_name)
  return ds_module.get_split(split_name, dataset_dir=FLAGS.dataset_dir)


def create_mparams():
  return {
      'conv_tower_fn':
      model.ConvTowerParams(final_endpoint=FLAGS.final_endpoint),
      'sequence_logit_fn':
      model.SequenceLogitsParams(
          use_attention=FLAGS.use_attention,
          use_autoregression=FLAGS.use_autoregression,
          num_lstm_units=FLAGS.num_lstm_units,
          weight_decay=FLAGS.weight_decay,
          lstm_state_clip_value=FLAGS.lstm_state_clip_value),
      'sequence_loss_fn':
      model.SequenceLossParams(
          label_smoothing=FLAGS.label_smoothing,
          ignore_nulls=FLAGS.ignore_nulls,
          average_across_timesteps=FLAGS.average_across_timesteps)
  }


def create_model(*args, **kwargs):
  ocr_model = model.Model(mparams=create_mparams(), *args, **kwargs)
  return ocr_model



        
    