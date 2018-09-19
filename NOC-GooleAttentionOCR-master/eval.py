# This code is modified from code implementing Google's attention based OCR. https://github.com/tensorflow/models/tree/master/research/attention_ocr
# Here for some specific usage in this project, some functions are added and some codes are modified. The original license is attached below for acknowledgement.
# ==============================================================================

# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import app
from tensorflow.python.platform import flags
import utils
import data_provider
import getopt
import os 
import system

split_name = 'test'
batch_size = 32
master = ''
num_batches = 100
eval_log_dir = os.path.abspath('.')
number_of_steps = None
eval_interval_secs = 60
train_log_dir = os.path.abspath('.')

def main(_):
  tmp_capStore = os.path.abspath('.') + 'caption.txt'
  opts, args = getopt.getopt(sys.argv[1:], "hi:o:")
  input_file=""
  output_file=tmp_capStore
  for op, value in opts:
      if op == "-i":
          input_file = value
      elif op == "-o":
          output_file = value
      elif op == "-h":
          usage()
          sys.exit()
  dataset = utils.create_dataset(split_name=split_name)
  model = utils.create_model(dataset.num_char_classes,
                                    dataset.max_sequence_length,
                                    dataset.num_of_views, dataset.null_code)
  data = data_provider.get_data(
      dataset,
      batch_size,
      augment=False,
      central_crop_size=utils.get_crop_size())
  endpoints = model.create_base(data.images, labels_one_hot=None)
  model.create_loss(data, endpoints)
  eval_ops = model.create_summaries(
      data, endpoints, dataset.charset, is_training=False)
  slim.get_or_create_global_step()
  session_config = tf.ConfigProto(device_count={"GPU": 0})
  slim.evaluation.evaluation_loop(
      master=master,
      checkpoint_dir=train_log_dir,
      logdir=eval_log_dir,
      eval_op=eval_ops,
      num_evals=num_batches,
      eval_interval_secs=eval_interval_secs,
      max_number_of_evaluations=number_of_steps,
      session_config=session_config)


if __name__ == '__main__':
  app.run()
