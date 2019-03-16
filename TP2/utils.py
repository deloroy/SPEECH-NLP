#https://github.com/edupoux/MVA_2019_SL/tree/master/TD_%232

#https://cs.brown.edu/courses/csci1460/assets/files/parsing.pdf

import pandas as pd
import numpy as np
import math

import pickle
from operator import itemgetter
import re

from copy import deepcopy

path_to_data = "data/"


def sentence(postag):
    postag_splitted = postag.split() #into a list
    sent = []
    for bloc in postag_splitted:
        if (bloc[0]=="("):
            continue
        else:
            word = ""
            for caract in bloc:
                if (caract==")"):
                    break
                word += caract
            sent.append(word)
    return ' '.join(sent)




def non_functional_tag(functional_tag):
    tag = ""
    for caract in functional_tag:
        if caract=="-":
            break
        tag+=caract
    return tag


def all_symbols(grammar):
    #replace by set
    res =  []
    for (root_tag,rules) in grammar.items():
        res.append(root_tag)
        for list_tags in rules.keys():
            for tag in list_tags:
                res.append(tag)
    return list(np.unique(res))



def add(dico, word, tag, counts = 1):
    # incrementing dico[word][tag], word is a string, tag is a string or a list (will be converted to a tuple in such case)

    if type(tag) == list:
        tag = tuple(tag)
    if word in dico.keys():
        if tag in dico[word].keys():
            dico[word][tag] += counts
        else:
            dico[word][tag] = counts
    else:
        dico[word] = {tag: counts}

def normalize_counts(dico):
    #convert counts to probabilities
    #ie: perform for each idx, the transformation below
    #dico[idx] = {i:c,j:d} ->  dico[idx] = {i:c/(c+d),j:d/(c+d)}

    res = deepcopy(dico)
    for (word,tags_counts) in dico.items():
        total_counts = np.sum(list(tags_counts.values()))
        for tag in tags_counts.keys():
            res[word][tag] /= total_counts
    return res




####################################################################################################
#the three functions below are imported from https://nbviewer.jupyter.org/gist/aboSamoor/6046170

def case_normalizer(word, dictionary):
    """ In case the word is not available in the vocabulary,
     we can try multiple case normalizing procedure.
     We consider the best substitute to be the one with the lowest index,
     which is equivalent to the most frequent alternative."""
    w = word
    lower = (dictionary.get(w.lower(), 1e12), w.lower())
    upper = (dictionary.get(w.upper(), 1e12), w.upper())
    title = (dictionary.get(w.title(), 1e12), w.title())
    results = [lower, upper, title]
    results.sort()
    index, w = results[0]
    if index != 1e12:
        return w
    return word

def normalize(word, word_id):
    DIGITS = re.compile("[0-9]", re.UNICODE)
    """ Find the closest alternative in case the word is OOV."""
    if not word in word_id:
        word = DIGITS.sub("#", word)
    if not word in word_id:
        word = case_normalizer(word, word_id)

    if not word in word_id:
        return None
    return word

def l2_nearest(embeddings, query_embedding, k):
    """Sorts words according to their Euclidean distance.
       To use cosine distance, embeddings has to be normalized so that their l2 norm is 1.
       indeed (a-b)^2"= a^2 + b^2 - 2a^b = 2*(1-cos(a,b)) of a and b are norm 1"""
    distances = (((embeddings - query_embedding) ** 2).sum(axis=1) ** 0.5)
    sorted_distances = sorted(enumerate(distances), key=itemgetter(1))
    return zip(*sorted_distances[:k])