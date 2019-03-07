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
        
def non_functional_tag(functional_tag):
    tag = ""
    for caract in functional_tag:
        if caract=="-":
            break
        tag+=caract
    return tag

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

print("building pcfg from corpus train")
print("")

#grammar, lexicon, counts_tokens = PGFG(corpus_train,normalized_counts=True,return_counts_tokens=True)
grammar_counts, lexicon_counts, counts_tokens = PGFG(corpus_train, normalized_counts=False,return_counts_tokens=True)
lexicon = normalize_counts(lexicon_counts)

#print(grammar_counts)
#print(counts_tokens)
#print(np.sum(list(grammar['COORD'].values())))
#print(grammar)
#print(grammar['SENT'])

set_terminal_symbols = all_terminal_symbols(lexicon)
nb_terminal_symbols = len(set_terminal_symbols)

set_all_symbols = all_symbols(grammar_counts)
nb_all_symbols = len(set_all_symbols)

print("list of all symbols")
print(set_all_symbols)
print("")

print("list of terminal symbols")
print(set_terminal_symbols)
print("")

print("##################################")
print("")

