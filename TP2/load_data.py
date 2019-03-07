from utils import *

file_corpus = open(path_to_data+"SEQUOIA_treebank","r")
corpus = []
for line in file_corpus:
    corpus.append(line)
file_corpus.close()

#print(corpus[0:10])


############################################################################

#Splitting corpus into train/dev/test set

frac_train = 0.8
frac_dev = 0.1
frac_test = 1-frac_train-frac_dev

N = len(corpus)
nb_train = int(N*frac_train)
nb_dev = int(N*frac_dev)

corpus_train = corpus[:nb_train]
corpus_dev = corpus[nb_train:nb_train+nb_dev]
corpus_test = corpus[nb_train+nb_dev:]

#print(N,nb_train,nb_dev)

