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

