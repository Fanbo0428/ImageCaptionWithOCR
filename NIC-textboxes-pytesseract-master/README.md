#Contents
1.[Introduction](#Introduction)
2.[Installation](#Installation)
3.[Test](#Test)
4.[Train](#Train)
5.[FAQs](#FAQs)


#Introductions
This directory contains code on the NIC+TextBoxes+Tesseract model mentioned in report. See the corresponding part of the report for details. 
#Installation
For running the code, make sure that you've got environment for Google's NIC model in your computer first. See https://github.com/tensorflow/models/tree/master/research/im2txt for details.

After you build up the enrironment for running Google's NIC model, copy the file 'run_inference.py' to replace the original one in the corresponding directory. 

For text recogniton tasks, 'TextBoxes' is utilized. See https://github.com/MhLiao/TextBoxes for installation details. Here author cite TextBoxes as the bibitex format given below by developers for TextBoxes:

    @inproceedings{LiaoSBWL17,
      author    = {Minghui Liao and
                   Baoguang Shi and
                   Xiang Bai and
                   Xinggang Wang and
                   Wenyu Liu},
      title     = {TextBoxes: {A} Fast Text Detector with a Single Deep Neural Network},
      booktitle = {AAAI},
      year      = {2017}
    }

After you successfully install 'TextBoxes' in your computer, copy file 'TextBoxes.py' to /path/to/yourTextBoxes/examples/TextBoxes to replace the file 'demo.py'
	
This code also relies on tesseract (https://pypi.org/project/pytesseract/), you can run the command below in your terminal

$shell sudo pip install pytesseract

In this project, text recognition tasks and image caption tasks focus on English only, so the corresponding model for text recognition in English for tesseract should be installed also.

For some NLP techniques used in this project, SpaCy (https://spacy.io/) is used, you can run the command below for installing SpaCy

$shell sudo pip install spacy

The for models and data in SpaCy, download all the models for NLP in English. 

$shell sudo python -m spacy.en.download all

It is noticeable that in the prcocess of developing the models, some configurations in code are only acceptable in developer's computer. Here for running in your own computer, you need to modify some codes in the files below:

performance_test.py
run_inference.py
TextBoxes.py

You can see the comment in the positions indicating on how you change the code for your own environment.

For users who first running the model, running the command below in your terminal.

$shell python ~/install_nltk_data.py

This helps you download the NLP dependencies needed by NIC model.
 
 
#Test
For running test, type the command below in you terminal:

$shell python ~/performance_test.py -i inputImage

#Train
Training in this case have two meanings:
1. You want to train the image caption (Google NIC) model with you own data: see the train part in  https://github.com/tensorflow/models/tree/master/research/im2txt for details. You can also get information on how authors train the model by my own data in the project report.
2. You want to train the TextBoxes model with your own data: see the train part in README.md on https://github.com/MhLiao/TextBoxes for details. Also you can get the details on the models trained by developer of TextBoxes.
#FAQs
Some problems faced by authors and the solutions are listed here for convenience.
1.For running the SpaCy model in English, you may need to add 

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

in your bashrc or profile file and run command 

$shell source ~/profile 

2. For Errors in python comes in form of 'No modules name ...', you can run command :

$shell sudo pip install ...

to install the corresponding modules in python for this project

3. It can be seen from the code that the TextBoxes model is based on Caffe (http://caffe.berkeleyvision.org/). For users who already have build up the environment of caffe or familiar with caffe, note that here in TextBoxes, CAFFE_ROOT need to be in the directory where you cloned the TextBoxes instead of the one you originally set up. You can see in the beginning of file 'TextBoxes.py':

caffe_root = '/Users/apple/Desktop/TextBoxes/'
os.chdir(caffe_root)
sys.path.append(0,'python')
import caffe

Here caffe is imported after the running directory is located into TextBoxes. It means that you don't need to install the caffe from original version, for doing the specific task of text detection with 'TextBoxes', you can just download the modified version caffe from github directory of 'TextBoxes'


For further detail and problems to discuss, you can refer to the project report and contact me by emails: fm2616@ic.ac.uk

