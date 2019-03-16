from utils import *

class PCFG():

    def __init__(self, corpus):

        self.grammar = {}
        self.lexicon = {}
        self.freq_tokens = {}

        #########################################################################
        self.extract_from_corpus(corpus)

        self.freq_tokens = {}
        for tag in self.lexicon.keys():
            for word in self.lexicon[tag].keys():
                if word in self.freq_tokens.keys():
                    self.freq_tokens[word] += self.lexicon[tag][word]
                else:
                    self.freq_tokens[word] = self.lexicon[tag][word]
        sum = np.sum(list(self.freq_tokens.values()))
        for word in self.freq_tokens:
            self.freq_tokens[word] /= sum
        #########################################################################
        self.binarize()

        self.freq_terminal_tags = {tag:np.sum(list(counts.values())) for (tag, counts) in self.lexicon.items()}
        sum = np.sum(list(self.freq_terminal_tags.values()))
        for tag in self.freq_terminal_tags:
            self.freq_terminal_tags[tag] /= sum
        #########################################################################

        self.normalize_counts()
        #########################################################################
        # storing the list of symbols (only tags, and with all artificial symbols)

        list_all_symbols = all_symbols(self.grammar)
        self.list_artificial_symbols = list(self.set_artificial_symbols)
        self.list_tags = list(set(list_all_symbols).difference(self.set_artificial_symbols))

        self.list_all_symbols = self.list_tags + self.list_artificial_symbols
        self.nb_tags = len(self.list_tags)
        self.nb_all_symbols = len(self.list_all_symbols)

    def extract_from_corpus(self, corpus):

        for postagged_sent in corpus:

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
                    add(self.lexicon, current_tag, word) #adding the pair (word,postag) to the lexicon
                    level -= nb_closing_brackets #since we closed a bracket

                    for k in range(nb_closing_brackets-1,0,-1): #at least 2 brackets closed -> new self.grammar rule defined
                        root = hierarchy[-2][-1] #root tag
                        if root=='': #if the root is the beginning of the sentence
                            break
                        tags = hierarchy[-1] #child tags
                        add(self.grammar, root, tags) #adding the rule to the grammar
                        hierarchy.pop() #popping from the hierarchy the childs list

                        #print(root,tags)
                        #print(hierarchy)

    def binarize(self):
        # self.grammar with counts(not probas) !!!
        # cf. https://en.wikipedia.org/wiki/Chomsky_normal_form

        # convert into Chomsky_normal_form

        # no need for START RULE (tag 'SENT' is already always at the left)
        # no need for TERM RULE (eliminate rules with nonsolitary terminals)

        self.set_artificial_symbols = set()

        # apply BIN RULE (eliminate right-hand sides with more than 2 nonterminals)
        self.apply_BIN_rule()

        # apply UNIT rule (eliminate unit rules)
        self.apply_UNIT_rule()

        # no need for DEL rules (no such cases)


    def normalize_counts(self):

        # convert counts into probabilities of grammar rules (from a given root) / words (for a given tag)
        self.grammar = normalize_counts(self.grammar)
        self.lexicon = normalize_counts(self.lexicon)

    ############################################################################
    #TO PUT GRAMMAR IN CHOMSKY NOMRAL FORM (called by binarize function)

    def apply_BIN_rule(self):
        #apply BIN RULE (eliminate right-hand sides with more than 2 nonterminals)

        grammar0 = deepcopy(self.grammar)

        for (root_tag, rules) in grammar0.items():
            #root_tag is the left hand symbol of the grammar rule
            #rules are the PCFC rules for derivation of root_tag

            for (list_tags, counts) in rules.items():
                nb_consecutive_tags = len(list_tags)

                if nb_consecutive_tags>2:
                    del self.grammar[root_tag][list_tags]
                    #print(list_tags)

                    symbol = root_tag + "|" + '-'.join(list_tags[1:])
                    self.set_artificial_symbols.add(symbol)
                    add(self.grammar, root_tag, (list_tags[0],symbol), counts=counts)
                    #print(root_tag, (list_tags[0],symbol))

                    for k in range(1,nb_consecutive_tags-2):
                        new_symbol = root_tag + "|" + '-'.join(list_tags[k+1:])
                        self.set_artificial_symbols.add(new_symbol)
                        add(self.grammar, symbol, (list_tags[k],new_symbol), counts=counts)
                        #print(symbol, (list_tags[k],new_symbol))
                        symbol = new_symbol

                    add(self.grammar, symbol, (list_tags[-2],list_tags[-1]), counts=counts)
                    #print(symbol, (list_tags[-2],list_tags[-1]))
                    #print("")

    def apply_UNIT_rule(self):

        grammar0 =  deepcopy(self.grammar)
        lexicon0 =  deepcopy(self.lexicon)

        #apply UNIT rule (eliminate unit rules)

        rules_to_remove = []

        for (root_tag, rules) in grammar0.items():
            #root_tag is the left hand symbol of the grammar rule
            #rules are the PCFC rules for derivation of root_tag

            for (list_tags, counts) in rules.items():

                if len(list_tags)==1: #unit rule A->B

                    child_tag = list_tags[0]
                    rules_to_remove.append((root_tag,list_tags))

                    freq = counts/(np.sum(list(self.grammar[root_tag].values())))

                    if child_tag in lexicon0.keys(): #existing rule A -> B where B is a preterminal symbol

                        symbol = root_tag + "&" + child_tag
                        self.set_artificial_symbols.add(symbol)

                        for (word, counts2) in lexicon0[child_tag].items(): #existing rule B -> word
                            add(self.lexicon, symbol, word, counts = counts2 * freq) #add A&B -> word, self.lexicon[word][A&B] = freq(A->B) * counts(B)

                        for (root_tag2, rules2) in grammar0.items():
                            for (list_tags2, counts2) in rules2.items():
                                if (len(list_tags2) == 2) and (list_tags2[1] == root_tag): #existing rule X -> Y A
                                    add(self.grammar, root_tag2, (list_tags2[0],symbol), counts=counts2) # add rule X -> Y A&B

                    else:   #existing rule A -> B where B is not a preterminal symbol
                        for (list_tags_child, counts2) in grammar0[child_tag].items():
                            if len(list_tags_child) == 2:  #existing rule B -> X1 X2
                                add(self.grammar, root_tag, list_tags_child, counts=counts2*freq) #add rule A -> X1 X2

        for (left, right) in rules_to_remove:
            del self.grammar[left][right]