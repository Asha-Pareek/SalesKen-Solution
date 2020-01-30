"""
Requirements : Gensim
pip install gensim
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import gensim.downloader as api
import json

## Download trained embedding.
model = api.load('glove-twitter-25')

def get_sentence_vector(sentence, model, available_words):
    """
    Returns a sentence vector by averaging vectors for each word in sentence.
    sentence : Given Sentence
    model : Embedding
    available_words : available Words in embedding
    """
    list_of_word_ = sentence.split()
    sent_vec = np.zeros((25,), dtype = 'float32')
    number_of_words = len(list_of_word_)
    for word in list_of_word_:
        if word in available_words:
            sent_vec = np.add(sent_vec, model[word])
    sent_vec = sent_vec/ (number_of_words + 0.0001)
    return sent_vec

def preprocess(list_of_sentences):
    """
    Given a list of sentences, returns a preprocessed list of sentences (Very basic preprocessing)
    """
    ret_list = []
    for f in list_of_sentences:
        f = f.lower()
        f= f.replace('\n', '')
        f= f.replace('?','')
        ret_list.append(f)
    return ret_list

def cosine_simil(a,b):
    """
    Given two vectors a & b, returns cosine similarity between a & b.
    """
    return np.linalg.multi_dot([a,b]) / (np.linalg.norm(a) * np.linalg.norm(b))


##Get available words.
available_words = set(model.vocab.keys())

## Load list of sentences
f = open('list_of_sentences')
lines = f.readlines()

##Preprocess lines
lines_mod = preprocess(lines)

## Get vector for each sentence in lines.
vector_lines = {x : get_sentence_vector(x, model, available_words) for x in lines_mod}

## For every sentence get list of sentences in descending order of semantic similarity.
semantic_sent = {}
for sent,vect in vector_lines.items():
    semantic_sent[sent] = [(x, cosine_simil(y, vect)) for x,y in vector_lines.items() if x!= sent]
    semantic_sent[sent].sort(key = lambda x : x[1], reverse = True)
    ## Comment this line to see the semantic similarity.
    semantic_sent[sent] = list(map(lambda x : x[0], semantic_sent[sent]))

## Write output to  json
with open("semantic_simil.json", 'w') as f:
    json.dump(semantic_sent, f)


