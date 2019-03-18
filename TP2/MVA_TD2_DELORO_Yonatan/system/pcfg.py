from utils import *

class PCFG():

    def __init__(self, corpus):

        self.grammar = {}
        self.lexicon = {}

        #########################################################################
        self.extract_from_corpus(corpus)
        # this function will fill:
        # -  self.grammar as a dictionary such that self.grammar[X] is a dictionary for each tag X
        # with X1...Xn as keys and counts(X -> X1...Xn) as values
        # - self.lexicon as a dictionary such that self.grammar[X] is a dictionary for each tag X
        # with words as keys and counts(X -> word) as values

        # frequencies of each word/token
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
        # this function introduce artificial symbols to put the grammar in Chomsky form
        self.binarize()

        #frequencies of each POS tag (ie a tag such that there exists a word st. tag -> word)
        self.freq_terminal_tags = {tag:np.sum(list(counts.values())) for (tag, counts) in self.lexicon.items()}
        sum = np.sum(list(self.freq_terminal_tags.values()))
        for tag in self.freq_terminal_tags:
            self.freq_terminal_tags[tag] /= sum

        #########################################################################
        #converting counts in self.grammar and self.lexicon into probabilities
        self.normalize_counts()

        #########################################################################
        #storing the list of symbols (only tags, and with all artificial symbols)

        list_all_symbols = all_symbols(self.grammar)
        self.list_artificial_symbols = list(self.set_artificial_symbols)
        self.list_tags = list(set(list_all_symbols).difference(self.set_artificial_symbols))

        self.list_all_symbols = self.list_tags + self.list_artificial_symbols
        self.nb_tags = len(self.list_tags)
        self.nb_all_symbols = len(self.list_all_symbols)

    def extract_from_corpus(self, corpus):

        #extract grammar and lexicon from corpus

        for tagged_sent in corpus:

            sent = tagged_sent.split() #into a list
            hierarchy = [] #index = number of opened brackets since the beginning of the sentence
                           #hierarchy[index] = list of tags pointed by root tag hierarchy[index-1]
            hierarchy.append([]) #list for level 0

            level = 0 #current difference between the number of opened brackets (minus the first one) and the number of closed brackets
            current_tag = None

            for bloc in sent:

                if (bloc[0]=="("): #then the bloc is introducing a new tag

                    tag = non_functional_tag(bloc[1:])  #we add it to the hierarchy
                    if level<len(hierarchy): #there is already one tag as its level
                        hierarchy[level].append(tag)
                    else: #first tag as its level
                        hierarchy.append([tag])
                    #print(hierarchy)
                    level += 1 #since we opened a new bracket
                    current_tag = tag #saved in order to add the word to the lexicon

                else: #then the bloc is introducing the word name and the number of closing brackets

                    word = ""
                    nb_closing_brackets = 0
                    for caract in bloc:
                        if (caract==")"):
                            nb_closing_brackets += 1
                        else:
                            word += caract
                    add(self.lexicon, current_tag, word) #adding the pair (word,tag) to the lexicon
                    level -= nb_closing_brackets #since we closed a bracket

                    for k in range(nb_closing_brackets-1,0,-1): #at least 2 brackets closed -> new self.grammar rule defined
                        root = hierarchy[-2][-1] #root tag
                        if root=='': #if the root is the beginning of the sentence
                            break
                        tags = hierarchy[-1] #child tags
                        add(self.grammar, root, tags) #adding the rule to the grammar
                        hierarchy.pop() #popping from the hierarchy the childs list

    def normalize_counts(self):
        # convert counts into probabilities of grammar rules (from a given root) / words (for a given tag)
        self.grammar = normalize_counts(self.grammar)
        self.lexicon = normalize_counts(self.lexicon)


    def binarize(self):
        # convert into Chomsky_normal_form, applying BIN and UNIT rule (the only one really necessary here)

        self.set_artificial_symbols = set() #set of artificial symbols introduced

        # apply BIN RULE (eliminate right-hand sides with more than 2 non-terminals)
        self.apply_BIN_rule()

        # apply UNIT rule (eliminate unit rules)
        self.apply_UNIT_rule()

    def apply_BIN_rule(self):
        #apply BIN RULE (eliminate right-hand sides with more than 2 nonterminals)

        grammar0 = deepcopy(self.grammar)

        for (root_tag, rules) in grammar0.items():
            #root_tag is the left hand symbol of the grammar rule

            for (list_tags, counts) in rules.items(): #list_tags in the righ hand term of the rule
                nb_consecutive_tags = len(list_tags)

                if nb_consecutive_tags>2:
                    del self.grammar[root_tag][list_tags]

                    symbol = root_tag + "|" + '-'.join(list_tags[1:])
                    self.set_artificial_symbols.add(symbol)
                    add(self.grammar, root_tag, (list_tags[0],symbol), counts=counts)

                    for k in range(1,nb_consecutive_tags-2):
                        new_symbol = root_tag + "|" + '-'.join(list_tags[k+1:])
                        self.set_artificial_symbols.add(new_symbol)
                        add(self.grammar, symbol, (list_tags[k],new_symbol), counts=counts)
                        symbol = new_symbol

                    add(self.grammar, symbol, (list_tags[-2],list_tags[-1]), counts=counts)

    def apply_UNIT_rule(self):
        # apply UNIT rule (eliminate unit rules)

        grammar0 =  deepcopy(self.grammar)
        lexicon0 =  deepcopy(self.lexicon)

        rules_to_remove = []

        for (root_tag, rules) in grammar0.items():
            #root_tag is the left hand symbol of the grammar rule

            for (list_tags, counts) in rules.items(): #list_tags in the righ hand term of the rule

                if len(list_tags)==1: #unit rule A->B

                    child_tag = list_tags[0]
                    rules_to_remove.append((root_tag,list_tags))

                    freq = counts/(np.sum(list(self.grammar[root_tag].values())))

                    if child_tag in lexicon0.keys(): #existing rule A -> B where B is a preterminal symbol

                        if root_tag!="SENT":
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