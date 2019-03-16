from utils import *
from utils_tree import postagged_sent_to_tree, tree_to_postagged_sent

from pcfg import PCFG
from oov import OOV

class CYK_Parser():

    def __init__(self, corpus_train):

        self.PCFG = PCFG(corpus_train)
        self.OOV = OOV(self.PCFG.lexicon, self.PCFG.list_all_symbols, self.PCFG.freq_tokens)

        # if the id of a symbol is above self.PCFG.nb_tags, its an artificial symbol added for normalization (not a gramm tag)
        self.symbol_to_id = {symbol: i for (i, symbol) in enumerate(self.PCFG.list_all_symbols)}

        self.lexicon_inverted = {word:{} for word in self.OOV.words_lexicon}
        for tag in self.PCFG.lexicon:
            for word in self.PCFG.lexicon[tag]:
                self.lexicon_inverted[word][tag] = self.PCFG.lexicon[tag][word]


    def compute_CYK_tables(self, sentence, viz_oov=False):
        # (cf. https://en.wikipedia.org/wiki/CYK_algorithm)
        # finding most likely symbol deriving each substring, for increasing length of substring (from 1 to length of the sentence)
        # and storing each time the position of the cut and the self.PCFG.grammar rule enabling to reach such most likely derivation

        nb_words = len(sentence)

        max_proba_derivation = np.zeros((nb_words, nb_words, self.PCFG.nb_all_symbols))
        # max_proba_derivation[s,l,a] is the maximum probability of
        # a parsing where symbol a derives substring x_s...x_(s+l)

        split_reaching_max = np.zeros((nb_words, nb_words, self.PCFG.nb_all_symbols, 3))
        # split_reaching_max[s,l,a,0] stores index cut
        # split_reaching_max[s,l,a,1] stores symbol b
        # split_reaching_max[s,l,a,2] stores symbol c

        # (i) b derives x_s...x_(s+cut), c derives x_(s+cut)...x_(s+l)
        # and a rewrites bc (a->bc in the self.PCFG.grammar)

        # (ii) the splitting <cut,b,c> defined by (i) is the one enabling
        # to reach the maximum probability for a to derives  x_s...x_(s+l)
        # (ie enabling to reach max_proba_derivation[s,l,a])

        # probabilities of tags for unary strings (words)
        for (position_word, word) in enumerate(sentence):

            token_to_tag = word

            if not(word in self.OOV.words_lexicon):
                if viz_oov: print(word+" is an OOV")
                token_to_tag = self.OOV.closest_in_corpus(word, viz_closest = viz_oov)
                if viz_oov:
                    if token_to_tag is None:
                        print("No closest token found")
                        print("")
                    else:
                        print("Closest token found : "+token_to_tag)
                        print("")

            if token_to_tag is None:
                for (tag,counts) in self.PCFG.freq_terminal_tags.items():
                    if tag in self.symbol_to_id:  # avoid the case where tag appearing in lexicon but not in grammar rules
                        id_tag = self.symbol_to_id[tag]
                        max_proba_derivation[position_word, 0, id_tag] = counts
            else:
                for (tag, proba) in self.lexicon_inverted[token_to_tag].items():
                    if tag in self.symbol_to_id: #avoid the case where tag appearing in lexicon but not in grammar rules
                        id_tag = self.symbol_to_id[tag]
                        max_proba_derivation[position_word, 0, id_tag] = proba

        # print(max_proba_derivation[:,0,:])

        for l in range(1, nb_words):
            # we will consider symbols deriving strings of length l+1...

            for s in range(nb_words - l):
                # ... and starting at index s of the sentence

                for cut in range(0, l):
                    # ... and such that the symbol can rewrite as two symbols AB
                    # with A deriving substring until index cut included, and B deriving substring from index cut+1

                    for (root_tag, rules) in self.PCFG.grammar.items():
                        # root_tag is the left hand symbol of the self.PCFG.grammar rule
                        # rules are the PCFC rules for derivation of root_tag

                        idx_root_tag = self.symbol_to_id[root_tag]

                        for (split, proba) in rules.items():
                            # root_tag can rewrite split[0]split[1] with probability proba

                            idx_left_tag = self.symbol_to_id[split[0]]  # idx of left split tag
                            idx_right_tag = self.symbol_to_id[split[1]]  # idx of right split tag

                            proba_decomposition = proba
                            proba_decomposition *= max_proba_derivation[s, cut, idx_left_tag]
                            proba_decomposition *= max_proba_derivation[s + cut + 1, l - cut - 1, idx_right_tag]

                            if proba_decomposition > max_proba_derivation[s, l, idx_root_tag]:
                                # therefore, we found a new decomposition <cut,split[0],split[1]>
                                # reaching a highest probability for root_tag to derive substring x_s...x_(s+l)

                                max_proba_derivation[s, l, idx_root_tag] = proba_decomposition
                                split_reaching_max[s, l, idx_root_tag, 0] = cut
                                split_reaching_max[s, l, idx_root_tag, 1] = idx_left_tag
                                split_reaching_max[s, l, idx_root_tag, 2] = idx_right_tag

            # print(max_proba_derivation[:,l,:])

        return max_proba_derivation, split_reaching_max.astype(int)


    # Rq for report : max_proba_derivation is non zero if there exists a triplet such that both are non zero and ...


    def parse_substring(self, s, l, idx_root_tag, sentence, max_proba_derivation, split_reaching_max):
        # parse substring beginning at index s of sentence, of length l+1, and tagged as idx_root_tag

        if l == 0:  # void string
            return sentence[s]

        else:  # split enabling to reach max_proba_derivation[s,l,idx_root_tag]
            cut = split_reaching_max[s, l, idx_root_tag, 0]
            idx_left_tag = split_reaching_max[s, l, idx_root_tag, 1]
            idx_right_tag = split_reaching_max[s, l, idx_root_tag, 2]

            left_tag = self.PCFG.list_all_symbols[idx_left_tag]
            right_tag = self.PCFG.list_all_symbols[idx_right_tag]

            # print(l,cut,l-cut)

            return [[left_tag, self.parse_substring(s, cut, idx_left_tag, sentence, max_proba_derivation, split_reaching_max)],
                    [right_tag, self.parse_substring(s + cut + 1, l - cut - 1, idx_right_tag, sentence, max_proba_derivation,
                                                split_reaching_max)]]

    def remove_artificial_symbols(self, T):
        #removing artificial symbols from T tree storing the parsing of the sentence

        #debinarize : remove artificial symbols of type X|X1X2X3 (from BIN rule)
        #merging children of an artificial symbol to its father
        nodes = deepcopy(T.nodes)
        for node in nodes:
            children = list(T.successors(node))
            if len(children)==0: pass
            elif len(children)==1 and len(list(T.successors(children[0]))) == 0: pass
            else:
                father = list(T.predecessors(node))
                if len(father)==0: root=node
                else:
                    symbol = T.nodes[node]["name"]
                    if (self.symbol_to_id[symbol] >= self.PCFG.nb_tags) and ("|" in symbol):  # artificial symbol from BIN rule
                        for child in T.successors(node):
                            T.add_edge(father[0],child)
                        #print("removed node : ",node,T.nodes[node]["name"])
                        T.remove_node(node)

        #add pre_terminal symbols : remove artificial symbols of type A&B (from UNIT rule)
        #decompositing A&B into two symbols A and B (A father of B father of word)
        max_id_node = np.max(T.nodes())
        nodes = deepcopy(T.nodes)
        for node in nodes: 
            #pas la root 
            children = list(T.successors(node))
            if len(children) == 0 or node==root: pass
            elif len(children) == 1 and len(list(T.successors(children[0]))) == 0:
                symbol = T.nodes[node]["name"]

                if (self.symbol_to_id[symbol] >= self.PCFG.nb_tags) and ("&" in symbol):  # artificial symbol from UNIT rule
                    word = children[0]

                    idx_cut = None
                    for (idx, c) in enumerate(symbol):
                        if c == "&":
                            idx_cut = idx

                    T.nodes[node]["name"] = symbol[:idx_cut]

                    idx_pre_terminal_node = max_id_node+1
                    T.add_node(idx_pre_terminal_node, name=symbol[idx_cut + 1:])
                    max_id_node += 1

                    T.remove_edge(node, word)
                    # print("removed edge : ", (node, word))
                    T.add_edge(node, idx_pre_terminal_node)
                    T.add_edge(idx_pre_terminal_node, word)



    def reformat_parsing(self, parsing):
        # converting parsing stored as a dictionnary into the required format (with nested brackets)

        if type(parsing) == str:
            return parsing

        else:
            string = ""
            for el in parsing:
                root_tag = el[0]
                parsing_substring = el[1]
                string = string + "(" + root_tag + " " + self.reformat_parsing(parsing_substring) + ")" + " "
            string = string[:-1]
            return string


    def parse(self, sentence, remove_artificial_symbols = True, viz_oov=False):
        sentence = sentence.split()

        nb_words = len(sentence)

        max_proba_derivation, split_reaching_max = self.compute_CYK_tables(sentence, viz_oov=viz_oov)

        idx_root_tag = self.symbol_to_id["SENT"]

        parsing_list = self.parse_substring(0, nb_words - 1, idx_root_tag, sentence, max_proba_derivation, split_reaching_max)

        if remove_artificial_symbols:
            T = postagged_sent_to_tree("( (SENT " +self.reformat_parsing(parsing_list)+"))", remove_after_hyphen=False)
            #nx.draw(T, labels=nx.get_node_attributes(T, "name"), arrows=False, pos=graphviz_layout(T, prog='dot'))
            #plt.show()
            self.remove_artificial_symbols(T)
            return tree_to_postagged_sent(T)

        else:
            return "( (SENT " + self.reformat_parsing(parsing_list) + "))" #res = parsing_dico


