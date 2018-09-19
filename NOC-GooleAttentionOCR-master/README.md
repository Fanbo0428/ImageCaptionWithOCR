# Contents

1.[Introduction](#introduction)
2.[Installation](#Installation)
3.[Test](#Test)
4.[Training](#Training)
5.[FAQs](#FAQs)
# Introduction
This code is based on NOC-recurrent (https://github.com/vsubhashini/noc) model developed by caffe and attention based OCR (https://github.com/tensorflow/models/tree/master/research/attention_ocr) model developed by Tensorflow. The two deep learning models running together to do the job which can generate the captions for natural image with text recognition. 
# Installation
For installation environment, you need to make sure that you've successfully installed caffe (http://caffe.berkeleyvision.org/) in your computer. For image caption models installation you can download the code and follow the instruction on  (https://github.com/vsubhashini/noc). For text recognition model, attention based OCR model developed by Google (https://github.com/tensorflow/models/tree/master/research/attention_ocr) is used here. You can find the information on installation the model and code in the url given. 

After you make sure the two models can be runned in your computer. You need to replace the two files given by this project to the original one. 

noc_captioner.py
eval.py

You can doing the job by running the command below:

$shell cd /path/to/noc-recurrent/examples/noc/
$shell cp noc_captioner.py noc_captioner_backup.py 
$shell mv /path/to/NOC+GooleAttentionOCR/noc_captioner.py /path/to/noc-recurrent/examples/noc/
$shell cd /path/to/attention_ocr/python/
$shell cp eval.py eval_backup.py
$shell mv /path/to/NOC+GooleAttentionOCR/eval.py /path/to/attention_ocr/examples/noc/


Or you can just do the way you like.....

Then modify the file performance_test.py, some code there need to be specified into your custermized directory. 
  
# Test
For testing the performance of this image caption system with text recognition, run the command below in your terminal (Note that here since NOC model reads a file as input for dealing with multiple images, so you need to write your image path name into a txt file.)
'''
$shell python performance_test.py -i input_image.txt
'''
# Training
Training in this case plays a significant role in promoting the performance. The reasons are:
1. For image caption model (NOC-model), the model performs good in reading particular objects, but for ordering the sentence or describing the contents, it is not better than NIC. So you can train the model with your own data to improve the performance. 
2. It is noticeable that for OCR models used in this project. It is based on French Street Name Dataset. So it is only compatible for French or languages come in forms similar with French. So if you want this model to recognize languages other than French, you can train the model with your own data. In this case, I only use the original model without training using myself. Since English is similar with French in the form.
# FAQs
For problems and questions, contact:
fm2616@ic.ac.uk
