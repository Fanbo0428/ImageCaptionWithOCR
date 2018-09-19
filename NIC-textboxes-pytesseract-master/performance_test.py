#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import getopt
# import re
import logging
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt

#English in this canse is set as default language
lang = 'eng'
#Some nouns that can have something written on them
target = 'board'
# current_directory=os.path.abspath(.)

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
               print 'help'
               sys.exit()
          elif opt in ('-i',"--ifile"):
               inputfile = arg
          elif opt in ("-o","--ofile"):
               outputfile = arg
     logging.info('input file: %s', inputfile)
     # print 'input file:', inputfile
     # print 'outputfile:', outputfile
     #command for running Google's NIC mode
     
     cmd_nic='python /Users/apple/Desktop/IndividualProject/image_caption/runned_model/Show_And_Tell/run_inference.py'
     
     #For development syplicity, text file for temporily storing output can be created in running directory.
     tmp_d= '/Users/apple/Desktop/tmp.txt' 
     #tmp_d=os.path.abspath(.)
     i = os.system(cmd_nic+' --input_files ' +inputfile+'>'+tmp_d)
     assert i ==0
     
     NIC_caption = []
     file_object=open(tmp_d,'rU')
     try:
         for line in file_object:
             # print(line)
             NIC_caption.append(line)
     finally:
         file_object.close()
         
     os.system('rm '+tmp_d)# delete the temporay file
     #Get the NIC caption
     caption = NIC_caption[1]
     
     tmp_i = 'cache/result.jpg'
     cmd_textbox= 'python TextBoxes.py -i '+inputfile +' -o '+ tmp_i
     # Using the modified version of textboxes to cutout the area with text.
     os.system(cmd_textbox)

     text = pytesseract.image_to_string(Image.open(tmp_i), lang=lang)
     
     if 'UNK' in caption:
         # If NIC model knows there is text in image
         caption.replace('UNK',text)
         final_caption=caption
         # print (caption)
    
     else:
         #NLP techniques used here
         nounsCaption=utils.find_noun(unicode(caption))
         nounsCaption=[str(c) for c in nounsCaption]
         # print(nounsCaption)
         best_candidate=utils.find_similiar(nounsCaption, target)
         final_caption=caption + ' with ' + text + ' on ' + best_candidate
     
     img = Image.open(inputfile)
     plt.imshow(img)
     plt.axis('off')
     plt.title(final_caption)
     plt.show()
     
     
if __name__ =="__main__":
     main(sys.argv[1:])
