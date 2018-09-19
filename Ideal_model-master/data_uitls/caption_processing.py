# -*- coding: utf-8 -*-

import os

candidates_dict = {}
captions_dict = {}
template_captions = {}

with open('output.txt','r') as f:
     text1 = f.readlines()

for t in text1:
    sep = t.index(':')
    captions_dict[t[:sep-1]] = t[sep+2:]
    if '<UNK>' in t[sep+2:]:
        template_captions[t[:sep-1]] = t[sep+2:].replace(' .\n','')

with open('candidates.txt','r') as f:
     text2 = f.readlines()

for t in text2:
    sep = t.index(':')
    candidates_dict[t[:sep-1]] = t[sep+2:]

# print candidates_dict


for caption in captions_dict:
    for candidate in candidates_dict:
        if caption == candidate:
            # print captions_dict[caption], candidates_dict[candidate]
            template_captions[caption] = captions_dict[caption].replace(' .\n',' ') + 'with ' + ' `` <UNK> <UNK> ''" ' + 'on ' + candidates_dict[candidate].replace('\n',' ')
            # print template_captions[caption], caption
with open('template_captions.txt','w+') as f:
    for tc in template_captions:
        string = str(tc)+' : '+template_captions[tc]+'.\n'
        f.write(string)


            
