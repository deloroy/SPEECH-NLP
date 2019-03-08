from utils import *

from load_data import corpus_train

# Extracting a PCGF from the training corpus
        
def add(dico, word, tag): 
    #incrementing dico[word][tag], word is a string, tag is a string or a list (will be converted to a tuple in such case)
    
    if type(tag)==list:
        tag = tuple(tag)
    if word in dico.keys():
        if tag in dico[word].keys():
            dico[word][tag]+=1
        else:
            dico[word][tag] = 1
    else:
        dico[word] = {tag:1}

############################################################################

def PGFG(corpus, normalized_counts=True, return_counts_tokens=False):
    
    #corpus is postagged
    
    grammar = {}
    lexicon = {}
    
    for postagged_sent in corpus:
        #print(postagged_sent)
        
        sent = postagged_sent.split() #into a list
        hierarchy = [] #index = number of opened brackets since the beginning of the sentence
                       #hierarchy[index] = list of tags pointed by root tag hierarchy[index-1]
        hierarchy.append([]) #list for level 0
        
        level = 0 #current difference between the number of opened brackets (minus the first one) and the number of closed brackets
        current_tag = None
        
        for bloc in sent:
            
            if (bloc[0]=="("): #then the bloc is introducing a new postag
                
                postag = non_functional_tag(bloc[1:])  #we add it to the hierarchy
                if level<len(hierarchy): #there is already one postag as its level
                    hierarchy[level].append(postag)
                else: #first postag as its level
                    hierarchy.append([postag])
                #print(hierarchy)
                level += 1 #since we opened a new bracket
                current_tag = postag #saved in order to add the word to the lexicon
                
            else: #then the bloc is introducing the word name and the number of closing brackets
                
                word = ""
                nb_closing_brackets = 0
                for caract in bloc:
                    if (caract==")"):
                        nb_closing_brackets += 1
                    else:
                        word += caract
                add(lexicon, word, current_tag) #adding the pair (word,postag) to the lexicon
                level -= nb_closing_brackets #since we closed a bracket
                
                for k in range(nb_closing_brackets-1,0,-1): #at least 2 brackets closed -> new grammar rule defined
                    root = hierarchy[-2][-1] #root tag
                    if root=='': #if the root is the beginning of the sentence
                        break
                    tags = hierarchy[-1] #child tags
                    add(grammar, root, tags) #adding the rule to the grammar
                    hierarchy.pop() #popping from the hierarchy the childs list
                    
                    #print(root,tags)
                    #print(hierarchy)
           
    #building a dictionnary computing counts of tokens (disregarding tags)
    if return_counts_tokens:
        counts_tokens = {word:np.sum(list(tags_counts.values())) for (word,tags_counts) in lexicon.items()}
        
    #convert counts into probabilities of grammar rules (from a given root) / tags (for a given word)
    if normalized_counts:
        grammar = normalize_counts(grammar)
        lexicon = normalize_counts(lexicon)

    if return_counts_tokens:
        return grammar, lexicon, counts_tokens
    else:
        return grammar, lexicon

############################################################################
#FOR NORMALIZATION

def apply_BIN_rule(grammar):
    #apply BIN RULE (eliminate right-hand sides with more than 2 nonterminals)
    
    grammar0 = deepcopy(grammar)
    
    max_idx_new_symbol = 0
    
    for (root_tag, rules) in grammar0.items():
        #root_tag is the left hand symbol of the grammar rule
        #rules are the PCFC rules for derivation of root_tag

        for (list_tags, counts) in rules.items():
            #print(list_tags)
            nb_consecutive_tags = len(list_tags)
            
            if nb_consecutive_tags>2:                
                del grammar[root_tag][list_tags]
                
                symbol = "NEW_" + str(max_idx_new_symbol)
                grammar[root_tag][(list_tags[0],symbol)] = counts
                max_idx_new_symbol += 1
                #print(root_tag,list_tags[0],symbol)
                
                for k in range(1,nb_consecutive_tags-2):
                    new_symbol = "NEW_" + str(max_idx_new_symbol)
                    grammar[symbol] = {(list_tags[k],new_symbol): counts}
                    max_idx_new_symbol += 1
                    #print(symbol,list_tags[k],new_symbol)
                    symbol = new_symbol
                #print(symbol,list_tags[-2],list_tags[-1])
                
                grammar[symbol] = {(list_tags[-2],list_tags[-1]): counts}

def apply_UNIT_rule(grammar, lexicon):
    
    grammar0 =  deepcopy(grammar)
    lexicon0 =  deepcopy(lexicon)
    
    #apply UNIT rule (eliminate unit rules)
    
    for (root_tag, rules) in grammar0.items():
        #root_tag is the left hand symbol of the grammar rule
        #rules are the PCFC rules for derivation of root_tag

        for (list_tags, counts) in rules.items():
            
            if len(list_tags)==1: #unit rule A->B
                
                child_tag = list_tags[0]
                #print(root_tag,child_tag)

                del grammar[root_tag][list_tags]
                
                freq = counts/(np.sum(list(grammar[root_tag].values())))
                
                for (word, tags) in lexicon0.items():
                    for (tag, counts_as_tag) in tags.items():
                        if tag == child_tag:
                            #print(lexicon[word])
                            lexicon[word][root_tag] = counts_as_tag * freq #lexicon[word][A] = freq(A->B) * counts(B)
                            #print(lexicon[word])
                            

def binarize_PCFG(grammar, lexicon):
    #grammar with counts(not probas) !!!
    #cf. https://en.wikipedia.org/wiki/Chomsky_normal_form
    
    #convert into Chomsky_normal_form
        
    #no need for START RULE (tag 'SENT' is already always at the left)
    #no need for TERM RULE (eliminate rules with nonsolitary terminals)
        
    #apply BIN RULE (eliminate right-hand sides with more than 2 nonterminals)
    apply_BIN_rule(grammar)
    
    #apply UNIT rule (eliminate unit rules)
    apply_UNIT_rule(grammar, lexicon)

    #no need for DEL rules (no such cases)

############################################################################

print("building pcfg from corpus train")
print("")

#grammar, lexicon, counts_tokens = PGFG(corpus_train,normalized_counts=True,return_counts_tokens=True)
grammar_counts, lexicon_counts, counts_tokens = PGFG(corpus_train, normalized_counts=False,return_counts_tokens=True)

binarize_PCFG(grammar_counts, lexicon_counts)

grammar = normalize_counts(grammar_counts)
lexicon = normalize_counts(lexicon_counts)

list_all_symbols = all_symbols(grammar)

#rewriting list_all_symbols to have first the tags of list_tags and then the artificial symbols
list_artificial_symbols = []
list_tags = []
for s in list_all_symbols:
    if s[:3]!="NEW":
       list_tags.append(s) 
    else:
       list_artificial_symbols.append(s) 

list_all_symbols = list_tags + list_artificial_symbols
nb_tags = len(list_tags)
nb_all_symbols = len(list_all_symbols)

#print("list of tags")
#print(list(list_tags))
#print("")

#print("list of all symbols")
#print(list(list_all_symbols))
#print("")
