from utils import *

from pcfg import grammar_counts, set_all_symbols
from oov import tagger

def binarize_PCFG_grammar(grammar):
    
    binary_grammar = deepcopy(grammar)
    #grammar with counts !!!
    
    #convert into Chomsky_normal_form
    #cf. https://en.wikipedia.org/wiki/Chomsky_normal_form
        
    #no need for START RULE (tag 'SENT' is already always at the left)
    #no need for TERM RULE (no nonsolitary terminals)
    
    #apply BIN RULE (eliminate right-hand sides with more than 2 nonterminals)
    
    max_idx_new_symbol = 0
    
    for (root_tag, rules) in grammar.items():
        #root_tag is the left hand symbol of the grammar rule
        #rules are the PCFC rules for derivation of root_tag

        for (list_tags, proba) in rules.items():
            #print(list_tags)
            nb_consecutive_tags = len(list_tags)
            
            if nb_consecutive_tags>2:                

                counts = binary_grammar[root_tag][list_tags]
                del binary_grammar[root_tag][list_tags]
                
                symbol = "NEW_" + str(max_idx_new_symbol)
                max_idx_new_symbol += 1
                binary_grammar[root_tag][(list_tags[0],symbol)] = counts
                #print(root_tag,list_tags[0],symbol)
                for k in range(1,nb_consecutive_tags-2):
                    new_symbol = "NEW_" + str(max_idx_new_symbol)
                    max_idx_new_symbol += 1
                    binary_grammar[symbol] = {(list_tags[k],new_symbol): counts}
                    #print(symbol,list_tags[k],new_symbol)
                    symbol = new_symbol
                #print(symbol,list_tags[-2],list_tags[-1])
                #print("")
                binary_grammar[symbol] = {(list_tags[-2],list_tags[-1]): counts}
    
    #no need for DEL or UNIT rules (no such cases)
    
    return binary_grammar, max_idx_new_symbol

######################################################################################################

if True: #test
    binary_grammar = {"SENT":{("GRP1","GRP2"):1},"GRP1":{("ADV","VERB"):1},"GRP2":{("ART","NOM"):1}}
    tagger = {"Il":{"ADV":1},"demande":{"VERB":1},"le":{"ART":1},"renvoi":{"NOM":1}}
    set_all_symbols = ["SENT","GRP1","GRP2","ADV","VERB","ART","NOM","A","PUNCT"]  #redefining variable
    nb_all_symbols = len(set_all_symbols) #redefining variable
    tag_to_idtag = {tag:i for (i,tag) in enumerate(set_all_symbols)}


else:
    binary_grammar_counts, max_idx_new_symbol = binarize_PCFG_grammar(grammar_counts)
    binary_grammar = normalize_counts(binary_grammar_counts)
    #print(binary_grammar)
    #print(binary_grammar['SENT'])
    #print(np.sum(list(binary_grammar['SENT'].values())))

    new_symbols = ["NEW_"+str(s) for s in range(max_idx_new_symbol)]
    set_all_symbols = list(set_all_symbols) + new_symbols  #redefining variable
    nb_all_symbols = len(set_all_symbols) #redefining variable

    tag_to_idtag = {tag:i for (i,tag) in enumerate(set_all_symbols)}

######################################################################################################


EPS = math.pow(10,-10)

def compute_CYK_tables(sentence):
    #(cf. https://en.wikipedia.org/wiki/CYK_algorithm)
    #finding most likely symbol deriving each substring, for increasing length of substring (from 1 to length of the sentence)
    #and storing each time the position of the cut and the grammar rule enabling to reach such most likely derivation
    
    nb_words = len(sentence)
   
    max_proba_derivation = np.zeros((nb_words,nb_words,nb_all_symbols))
    #max_proba_derivation[s,l,a] is the maximum probability of
    #a parsing where symbol a derives substring x_s...x_(s+l)
    
    split_reaching_max = np.zeros((nb_words,nb_words,nb_all_symbols,3))
    #split_reaching_max[s,l,a,0] stores index cut
    #split_reaching_max[s,l,a,1] stores symbol b
    #split_reaching_max[s,l,a,2] stores symbol c
    
    #(i) b derives x_s...x_(s+cut), c derives x_(s+cut)...x_(s+l)
    #and a rewrites bc (a->bc in the grammar)
    
    #(ii) the splitting <cut,b,c> defined by (i) is the one enabling
    #to reach the maximum probability for a to derives  x_s...x_(s+l)
    #(ie enabling to reach max_proba_derivation[s,l,a])

    #probabilities of tags for unary strings (words)
    for (position_word,word) in enumerate(sentence):
        tags = tagger[word] #tagger(word)
        for (tag, proba) in tags.items():
            id_tag = tag_to_idtag[tag]
            max_proba_derivation[position_word,0,id_tag] = np.log(proba + EPS)
            
    for l in range(1, nb_words):
        #we will consider symbols deriving strings of length l+1...
        
        for s in range(nb_words-l):
            #... and starting at index s of the sentence
            
            for cut in range(1,l):
                #... and such that the symbol can rewrite as two symbols AB
                #with A deriving substring until index cut, and B deriving substring after index cut
                
                for (root_tag, rules) in binary_grammar.items():
                    #root_tag is the left hand symbol of the grammar rule
                    #rules are the PCFC rules for derivation of root_tag
                    
                    idx_root_tag = tag_to_idtag[root_tag]
                    
                    for (split, proba) in rules.items():
                        #root_tag can rewrite split[0]split[1] with probability proba
                        
                        if len(split)==2: #disregard rules A->B, consider only A->BC
                            
                            idx_left_tag = tag_to_idtag[split[0]] #idx of left split tag
                            idx_right_tag = tag_to_idtag[split[1]] #idx of right split tag

                            proba_decomposition = np.log(proba + EPS)
                            proba_decomposition += np.log(max_proba_derivation[s,cut,idx_left_tag] + EPS)
                            proba_decomposition += np.log(max_proba_derivation[s+cut,l-cut,idx_right_tag] + EPS)

                            if proba_decomposition > max_proba_derivation[s,l,idx_root_tag]:
                                #therefore, we found a new decomposition <cut,split[0],split[1]>
                                #reaching a highest probability for root_tag to derive substring x_s...x_(s+l)

                                max_proba_derivation[s,l,idx_root_tag] = proba_decomposition
                                split_reaching_max[s,l,idx_root_tag,0] = cut
                                split_reaching_max[s,l,idx_root_tag,1] = idx_left_tag
                                split_reaching_max[s,l,idx_root_tag,2] = idx_right_tag
                            
    return max_proba_derivation, split_reaching_max.astype(int)

#Rq for report : max_proba_derivation is non zero if there exists a triplet such that both are non zero and ...


def parse_substring(s,l,idx_root_tag, sentence, max_proba_derivation, split_reaching_max):
    #parse substring beginning at index s of sentence, of length l+1, and tagged as idx_root_tag
    
    nb_words = max_proba_derivation.shape[0]
    
    if l==0: #void string
        root_tag = set_all_symbols[idx_root_tag]
        
        return (root_tag, sentence[s])
    
    else: #split enabling to reach max_proba_derivation[s,l,idx_root_tag]
        cut = split_reaching_max[s,l,idx_root_tag,0]
        idx_left_tag = split_reaching_max[s,l,idx_root_tag,1]
        idx_right_tag = split_reaching_max[s,l,idx_root_tag,2]
        
        left_tag = set_all_symbols[idx_left_tag]
        right_tag = set_all_symbols[idx_right_tag]
        
        print(l,cut,l-cut)
            
        return {left_tag: parse_substring(s, cut, idx_left_tag, sentence, max_proba_derivation, split_reaching_max),
                right_tag: parse_substring(s+cut, l-cut, idx_right_tag, sentence, max_proba_derivation, split_reaching_max)}
        
def remove_artificial_symbols(parsing_dico):
    if type(parsing_dico)==tuple:
        return parsing_dico
    else:
        new_parsing_dico = {}
        for (root_tag,rules) in parsing_dico.items():
            if tag_to_idtag[root_tag]>=nb_tags: #artificial symbol
                dico = remove_artificial_symbols(rules)
                for (k2,v2) in dico.items():
                    new_parsing_dico[k2] = v2
            else:
                new_parsing_dico[root_tag] = rules
        return new_parsing_dico
            
def parse(sentence):

    sentence = sentence.split()

    nb_words = len(sentence)
    
    max_proba_derivation, split_reaching_max = compute_CYK_tables(sentence)    
        
    #idx_root_tag = np.argmax(max_proba_derivation[0,nb_words,:])
    idx_root_tag = tag_to_idtag["SENT"]
    #rq ca devrait etre toujours S_0 à ce point !!!
    
    parsing_dico = parse_substring(0,nb_words-1,idx_root_tag, sentence, max_proba_derivation, split_reaching_max)
    
    res = remove_artificial_symbols(parsing_dico)
    
    return res

def reformat_parsing(parsing):
    #converting parsing stored as a dictionnary into the required format (with nested brackets)
    
    if type(parsing)==tuple:
        tag = parsing[0]
        word = parsing[1]
        return "(" + tag + " " + word + ")"

    else:
        string = "("
        for (root_tag,parsing_substring) in parsing.items():
            string = string + "(" + root_tag + " " + reformat_parsing(parsing_substring) + ")" + " "
        string = string + ")"
        return string    

def parsing(sentence):
    return reformat_parsing(parse(sentence))

if True:
    # sentences_test = [sentence(postag) for postag in corpus_test]
    sent = "Il demande le renvoi"
    print(sent)
    print(parsing(sent))

