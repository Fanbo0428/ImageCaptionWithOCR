#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import unicode_literals

import os
import spacy

# with os.system('source /etc/profile'):
#     import spacy # For bug-fixing on problems related to 'unknown locale: UTF-8'

test_str = 'Yaniga is eating buns and drinking a bottle of water. '
original_word_count = '/Users/apple/Desktop/NIC+SWT+pytesseract/word_counts.txt'

def find_noun(sentence):
    result = list([])
    nlp = spacy.load('en')
    for token in nlp(sentence):
        result.append([token, token.pos_])
    nouns=list([])
    for w in result:
        if w[1] == 'NOUN':
            nouns.append(w[0])
    return nouns

##########################################
####The functions below are dealing with world dictionaries#####
##########################################

def find_all_nouns():#Find all nouns in the dictionary
    nlp = spacy.load('en')
    f = open(original_word_count,'r')
    result = list()
    for line in open(original_word_count,'r'):
        line = f.readline()
        line = ''.join([i for i in line if not i.isdigit()])
        # result.append(line)
        for token in nlp(line):
            if token.pos_=='NOUN':
                result.append(str(token))
            else:
                break
    # print result
    f.close()                
    open('nouns.txt', 'w').write('%s' % '\n'.join(result))
    
def find_similiar(words, target):
    #input a list of words for finding words that similiar to them
    nlp=spacy.load('en_core_web_lg')
    words = nlp(unicode(' '.join(words)))
    target=nlp(unicode(target))
    similiarity=list()
    for token in words:
        similiarity.append(token.similarity(target))
    # return results
    max_sim=max(similiarity)
    for token in words:
        if  token.similarity(target) == max_sim:
            return str(token)
            
# Comment after execute once
# find_all_nouns()
# print find_similiar(['apple','cat','dog'],'horse')

    
