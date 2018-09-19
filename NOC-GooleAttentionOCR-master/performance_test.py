# !/usr/bin/python
# -*- coding: UTF-8 -*-
#This is for testing the performance of NOC model with Google's attention based OCR model
import os 
import sys, getopt
import spacy
from PIL import Image
import matplotlib.pyplot as plt

def usage():
    #This is for instruction of how to operate the assembling model
    print('-i: inputImage List (stored in a file)')
    print('-o: outputFile')

#A list to store captions, and find the bet captions
captions = list()

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

#need to modify the code to where you save the file
noc_cmd='python ~/noc_captioner.py -i' + ' ' + input_file +'>' + ' ' +tmp_capStore
# run the noc image caption model

oc.system(noc_cmd)
with open(tmp_capStore) as f:
    captions = f.readlines()

#Operations for extracting the captions in the output of noc model
caption = captions[len(captions)-1]
index = caption.find('. ')
caption = caption[:index+1]

#Remove the temporary file for storing captions 
os.cmd('rm '+tmp_capStore)

nlp = spacy.load('en')
target =nlp(u'borad')
# This is for finding the nousn in the caption, for the sentence ordering
tokens = nlp(unicode(caption))

nousCaption={}
for token in tokens:
    if token.pos_ == 'NOUN':
        nounsCaption.append(str(token) : token.similarity(target))
        
# This gives us the noun which is most likely containing the text        
text_object = max(zip(nousCaption.values(),nousCaption.keys()))

#
# Text object: something in the image containing text
# Caption: Caption for image

#Then for deploying the Attention based OCR model:

os.system('python ~/attention-ocr/python eval.py -i '+ input_file + '> ' +os.path.abspath() + '/' + tmp_text) 
#Here input_file is the path to image.
os.chdir(os.path.abspath())#make sure the path remaining on the original one.
text_in_image =''
with open(tmp_text) as f:
    text_in_image = f.read()
    
os.system('rm '+tmp_text)#delete the temporary file.

final_caption = caption + 'with ' +text_in_image + 'on '+ text_object

img = Image.open(inputfile)
plt.imshow(img)
plt.axis('off')
plt.title(final_caption)
plt.show()














