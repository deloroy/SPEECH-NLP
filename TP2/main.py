from utils import *

from load_data import * #load data 
from pcfg import * #build pcfg
from oov import * #load embeddings, and build tagger for oov


print("test of the oov tagger")
print("")

sentences_test = [sentence(postag) for postag in corpus_test]
print(sentences_test[0])
print("")

for word in sentences_test[0].split():
    print(word, tagger(word,viz_closest = True))
    print("")
print("")

print("##################################")
print("")

