# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import spacy
import os
import re
import collections

def IsNoun(word):
    nlp=spacy.load('en')
    return nlp(unicode(word))[0].pos_ =='NOUN'

text=''
nounFrequency={}

with open('/Users/apple/Desktop/Ideal_model/intermediate_files/output.txt') as f:
    for caption in f:
        sep = caption.index(':')
        description = caption[sep+1:]
        text = text + description
#print text
frequency = {}
for word in text.split():
    if word not in frequency:
        frequency[word] = 1
    else:
        frequency[word] += 1

for word in frequency:
    if IsNoun(word):
        nounFrequency[word] = frequency[word]

nounFrequency = sorted(nounFrequency.items(),key=lambda item:item[1])
with open('/Users/apple/Desktop/noun_rank.txt','w') as f:
    for n in nounFrequency:
        string=str(n[0])+' : ' + str(n[1]) + '\n'
        f.write(string)
print('Done')
# print nounFrequency

