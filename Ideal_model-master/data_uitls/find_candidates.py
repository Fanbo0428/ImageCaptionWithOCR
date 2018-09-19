# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import spacy
import os
import re

# use a dict to store all the image names and corresponding caption
image_caption={}
candidates={}
def find_similar(words, target):
    #input a list of words for finding words that similiar to them
    nlp=spacy.load('en_core_web_lg')
    # words = nlp(unicode(' '.join(words)))
    target=nlp(unicode(target))
    similiarity=list()
    for token in words:
        token = nlp(unicode(token))
        similiarity.append(token.similarity(target))
    # return results
    max_sim=max(similiarity)
    for token in words:
        token = nlp(unicode(token))
        if  token.similarity(target) == max_sim:
            return str(token)
            
def find_noun(sentence):
    result = list([])
    nlp = spacy.load('en')
    for token in nlp(unicode(sentence)):
        result.append([token, token.pos_])
    nouns=list([])
    for w in result:
        if w[1] == 'NOUN':
            nouns.append(w[0])
    return nouns

with open('/Users/apple/Desktop/Ideal_model/intermediate_files/output.txt') as f:
    for caption in f:
        sep = caption.index(':')
        image_name = caption[:sep-1]
        description = caption[sep+1:]
        # print (image_name,description)
        image_caption[image_name] = description

# print len(image_caption), image_caption

count = 0
for fn in image_caption:
    sent = image_caption[fn]
    # print sent.split()
    if 'UNK' in sent:
        pass
    else:
        list_noun = find_noun(sent)
        candidates[fn] =  find_similar(list_noun,'text')
        print 'find candidate in '+ fn +' : '+ candidates[fn]
        count +=1

print('Total '+str(count)+' sentences have found candidates')

with open('/Users/apple/Desktop/Ideal_model/intermediate_files/candidates.txt','w') as wf:
    for i in candidates:
        wf.write(str(i) + ' : ' + candidates[i])
        
print('Candidates are written in to file: /Users/apple/Desktop/Ideal_model/intermediate_files/candidates.txt')

    