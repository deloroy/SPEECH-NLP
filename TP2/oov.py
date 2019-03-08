# Handling OOV (out of vocabulary words)

from utils import * 

from pcfg import lexicon, counts_tokens, list_all_symbols, nb_all_symbols

####################################################################################################
####################################################################################################

## PART A. To find the closest neighbor in the postags corpus of a word having an embedding


#the three functions below are imported from https://nbviewer.jupyter.org/gist/aboSamoor/6046170

# Noramlize digits by replacing them with #
DIGITS = re.compile("[0-9]", re.UNICODE)

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

####################################################################################################

def build_embeddings_lexicon(words_lexicon, words_with_embeddings_id, embeddings):
    # function returning the embeddings matrix of the words of lexicon having one
    # and the mapping id-word / word-id for these lexicon words

    words_lexicon_in_corpus = [] #words of lexicon having an embedding

    # Embeddings of words of lexicon present in embeddings corpus
    embeddings_lexicon = None
    for word in words_lexicon:
        word = normalize(word, words_with_embeddings_id)
        if not(word is None):
            words_lexicon_in_corpus.append(word)
            word_index = words_with_embeddings_id[word]
            id_word = words_with_embeddings_id[word]
            if embeddings_lexicon is None:
                embeddings_lexicon = embeddings[id_word]
            else:
                embeddings_lexicon = np.vstack([embeddings_lexicon,embeddings[id_word]])

    # Map lexicon words present in embedding corpus to new ids and vice versa
    word_lexicon_id = {w:i for (i, w) in enumerate(words_lexicon_in_corpus)}
    id_word_lexicon = words_lexicon_in_corpus
    
    return embeddings_lexicon, word_lexicon_id, id_word_lexicon

####################################################################################################

words_with_embeddings, embeddings = pickle.load(open(path_to_data+'polyglot-fr.pkl', "rb"), encoding='bytes') #or "bytes" or latin1"
words_with_embeddings_id = {w:i for (i, w) in enumerate(words_with_embeddings)}   # Map words to indices
words_lexicon = list(lexicon.keys())

embeddings_lexicon, word_lexicon_id, id_word_lexicon = build_embeddings_lexicon(words_lexicon, words_with_embeddings_id, embeddings)
embeddings_lexicon /= np.linalg.norm(embeddings_lexicon,axis=1)[:,None]


print("building oov tagger")
print("")

print(len(words_lexicon)," words in lexicon")
print(len(id_word_lexicon), " words in lexicon having an embedding")
print(embeddings_lexicon.shape)
#print(id_word_lexicon[0:2])
print("")

print("##################################")
print("")

####################################################################################################

def closest_word_in_corpus(query):
    #return nearest_neighbor(query, embeddings, word_id, id_word, 3)    
    query = normalize(query,words_with_embeddings_id)
    if not query:
        print("OOV word")
        return None
    query_index = words_with_embeddings_id[query]
    query_embedding = embeddings[query_index]
    indices, distances = l2_nearest(embeddings_lexicon, query_embedding, 1)
    neighbors = [id_word_lexicon[idx] for idx in indices]
    return neighbors[0]



####################################################################################################
####################################################################################################


## PART B. To find the closest neighbor in the vocabulary (postags corpus + embeddings corpus) of a word, considering spelling errors


vocabulary = list(words_with_embeddings) + words_lexicon
word_vocab_to_id = {w:i for (i, w) in enumerate(vocabulary)}

####################################################################################################

def levenstein_damerau_distance(word,word2):
    
    size_word2 = len(word2)
    dist = np.zeros((3,size_word2+1))
    #dist[0,j] = distance from word[:t-1] to word2[:j-1]
    #dist[1,j] = distance from word[:t] to word2[:j-1]
    #dist[2,j] = distance from word[:t+1] to word2[:j-1]
    #where w[:-1] is the void string
    #and where t worths 0 at the beginning and progressively increases up to size_word1-1
    #(enables to reach a linear space complexity, 
    #I do not save the whole matrix of distances between prefixes but only the last two lines)
    
    dist[0,:] = np.arange(size_word2+1) #distance from void string to word2[:j]
    dist[1,0] = 1
    for j in range(1,size_word2+1):
        diff_last_letters =  word[0]!=word2[j-1] #different last letters of prefixes
        dist[1,j] = min([dist[0][j]+1,dist[1][j-1]+1,dist[0][j-1]+diff_last_letters]) 
    
    for i in range(2,len(word)+1):
        
        dist[2][0] = i #distance from word[:i] to void string
        for j in range(1,size_word2+1):
            diff_last_letters =  word[i-1]!=word2[j-1] 
            dist[2,j] = min([dist[1][j]+1,dist[2][j-1]+1,dist[1][j-1]+diff_last_letters]) 
            if j>1: #consider swap too ! 
                if (word[i-1]==word2[j-2])and(word[i-2]==word2[j-1]):
                    dist[2,j] = min(dist[2,j],dist[0,j-2]+1)
        
        dist[0,:] = dist[1,:]
        dist[1,:] = dist[2,:]
        
    return dist[2][size_word2]

####################################################################################################

def corrected_word(query):
    
    #TODO : normalize query, and also treebank lexicon words !!!
    #query = DIGITS.sub("#", query)
    #query = case_normalizer(query, word_id)
    
    #le mot a une longueur différente : éliminer les cas ...
    
    candidates = {1:[],2:[],3:[]} #words at distances 1,2,3 from real words
    min_dist = 3 #distance with closest word
    
    for word in words_lexicon: 
        #we look for corrections in treebank corpus at most distance min_dist from query
        
        dist = levenstein_damerau_distance(query,word)
        if dist<=min_dist:
            candidates[dist].append(word)
            min_dist = dist
    
    if len(candidates[1])>0: 
        #there is at least one word in treebank corpus at distance 1, 
        #we return the most frequent of these
        
        idx_most_frequent = np.argmax([counts_tokens[word] for word in candidates[1]])
        return candidates[1][idx_most_frequent]
    
    #####################################
    #if we reached this line
    #all words in treebank corpus are at distance more than 2
    #we look for corrections in embeddings corpus at most distance min_dist from query

    for word in words_with_embeddings:
        dist = levenstein_damerau_distance(query,word)
        if dist<=min_dist:
            candidates[dist].append(word)
            min_dist = dist
        if min_dist==1: 
            #since no word at distance 1 was found previously in treebank corpus,
            #we return the word which has an embedding and accomplished distance 1 with query
            return candidates[1][0]

    #####################################
    #if we reached this line,
    #we found words in treebank/embeddings at at least distance 2 from the query
    
    list_candidates = candidates[min_dist]
    candidates_in_lexicon = []

    for word in list_candidates:
        if word in words_lexicon: candidates_in_lexicon.append(word)

    if len(candidates_in_lexicon)==0: 
        #the min distance is accomplished only by words which have an embedding, we return one of these
        return list_candidates[0]

    #####################################
    #if we reached this line
    #the min distace is accomplished by a word in treebank corpus, we return the most frequent of these
    idx_most_frequent = np.argmax([counts_tokens[word] for word in candidates_in_lexicon])
    return candidates_in_lexicon[idx_most_frequent]            



####################################################################################################
####################################################################################################

## PART C. Define tagger for OOV words

def tagger_oov(oov_word, viz_closest = False):

    if oov_word in words_with_embeddings:
        #look for words of corpus whose embedding is closest to the 
        #embedding of oov_word
        closest_corpus_word = closest_word_in_corpus(oov_word) #, embeddings, words_lexicon
        if viz_closest: print(closest_corpus_word, " is the closest word (meaning) found among lexicon words having an embedding")
        return lexicon[closest_corpus_word]
    
    else: #look for spelling errors
        correction = corrected_word(oov_word)
        
        if correction is None:
            if viz_closest: print("no corrected word (spelling) found at damerau-levenshtein distance less than 3")
            return {symbol:1/nb_all_symbols for symbol in list_all_symbols}
        
        else:  
            if viz_closest: print(correction, " is the closest word (spelling) found among words in the lexicon or having an embedding")

            if correction in words_lexicon: #if corrected word in corpus
                if viz_closest: print(correction, " is a word in the lexicon")
                return lexicon[correction]

            else: #if corrected word in embedding corpus
                closest_corpus_word = closest_word_in_corpus(correction) #, embeddings, words_lexicon
                if viz_closest: print(closest_corpus_word, " is the closest word (meaning) found among lexicon words having an embedding")
                return lexicon[closest_corpus_word]


def tagger(word, viz_oov = False):
    if word in lexicon:
        return lexicon[word]
    else:
        if viz_oov: print(word," is an OOV")
        return tagger_oov(word, viz_closest = viz_oov)


