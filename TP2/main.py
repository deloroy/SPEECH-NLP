from utils import *
from utils_tree import *

from cyk import CYK_Parser

file_corpus = open(path_to_data+"SEQUOIA_treebank","r")
corpus = []
for line in file_corpus:
    corpus.append(line)
file_corpus.close()
#print(corpus[0:10])

#option :
#see statistics
#number of sentences of corpus text to see
#see pylab score and execution time
#see management of oov rules for given sentence
#plot tree without removal of artificial symbols
#plot tree for given sentence
#save results in txt


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

print(str(len(corpus_test)) + " sentences in test corpus")

#Building Parser
print("Build CYK parser")
tic = time.time()
my_CYK_parser = CYK_Parser(corpus_train)
tac = time.time()
print("Done in "+str(round(tac-tic,2))+"s")

print("")
print("Some statistics")
#print(my_CYK_parser.PCFG.list_all_symbols[:my_CYK_parser.PCFG.nb_tags])
nb_tags = len(my_CYK_parser.PCFG.list_all_symbols[:my_CYK_parser.PCFG.nb_tags])
nb_total = len(my_CYK_parser.PCFG.list_all_symbols)
nb_artificial = nb_total-nb_tags
nb_artificial_bin = len([r for r in my_CYK_parser.PCFG.list_all_symbols[my_CYK_parser.PCFG.nb_tags:] if "|" in r])
print(nb_tags, " grammatical tags")
print(nb_artificial, " artificial symbols")
print(nb_artificial_bin, " artificial symbols from BIN rule")
print(nb_artificial-nb_artificial_bin, " artificial symbols from UNIT rule")
print(nb_total, " symbols (tags and artificial ones)")
print(len(my_CYK_parser.OOV.words_lexicon), " words in lexicon")
print(len(my_CYK_parser.OOV.Meaning_Search_Engine.id_word_lexicon), " words in lexicon having an embedding")
print(len(my_CYK_parser.OOV.words_with_embeddings), " words having an embedding")
#print(my_CYK_parser.Tagger.Meaning_Search_Engine.embeddings_lexicon.shape)


print("##############################")

gold_parsings = []
my_parsings = []

for (idx_sentence,human_parsing) in enumerate(corpus_test[len(corpus_test)-99]):

    print(human_parsing)

    T = postagged_sent_to_tree(human_parsing, remove_after_hyphen=True)
    gold_parsing = tree_to_postagged_sent(T) #removing functions in tags (parts after hyphen)

    print("")
    print("Sentence N°" + str(idx_sentence))
    print("")

    sent = sentence(gold_parsing)
    print(sent)
    print("")

    print("Gold Parsing")
    print(gold_parsing)
    print("")

    print("My Parsing")
    tic = time.time()
    my_parsing = my_CYK_parser.parse(sent, remove_artificial_symbols=True, viz_oov = False)
    print(my_parsing)
    tac = time.time()
    print("Done in " + str(round(tac - tic, 2)) + "s")
    print("")

    if False:
        from utils_tree import draw_tree
        draw_tree(my_parsing)

    gold_parsing = gold_parsing[2:-1]  # EVALPB works if sentence begins by (SENT and ends with )
    my_parsing = my_parsing[2:-1]  # EVALPB works if sentence begins by (SENT and ends with )
    gold_parsings.append(gold_parsing)
    my_parsings.append(my_parsing)

    print("Score PYEVALB :")
    gold_tree = PYEVALB_parser.create_from_bracket_string(gold_parsing)
    test_tree = PYEVALB_parser.create_from_bracket_string(my_parsing)
    result = PYEVALB_scorer.Scorer().score_trees(gold_tree, test_tree)
    print('Tagging accurracy ' + str(result.tag_accracy))

    print("##############################")

    if idx_sentence%3==0:

        with open('gold_parsings.txt', 'w') as f:
            for item in gold_parsings:
                f.write("%s\n" % item)
        with open('my_parsings.txt', 'w') as f:
            for item in my_parsings:
                f.write("%s\n" % item)

        PYEVALB_scorer.Scorer().evalb('gold_parsings.txt','my_parsings.txt', 'results.txt')




'''
if False:  # test cyk
    Parser = CYK_Parser(corpus_train)
    my_CYK_parser.PCFG.grammar = {"SENT": {("GRP1", "GRP2"): 1}, "GRP1": {("ADV", "VERB"): 1},
                                "GRP2": {("ART", "NOM"): 1}}
    tagger = {"Il": {"ADV": 1}, "demande": {"VERB": 1}, "le": {"ART": 1}, "renvoi": {"NOM": 1}}
    list_all_symbols = ["SENT", "GRP1", "GRP2", "ADV", "VERB", "ART", "NOM", "PUNCT"]  # redefining variable
    my_CYK_parser.PCFG.nb_tags = 8
    my_CYK_parser.PCFG.nb_all_symbols = len(list_all_symbols)  # redefining variable
    my_CYK_parser.symbol_to_id = {tag: i for (i, tag) in enumerate(list_all_symbols)}

else:
    for (root, rules) in my_CYK_parser.PCFG.grammar.items():
        for (right, proba) in rules.items():
            # print(root, right, proba)
            if len(right) != 2:
                print("Found a rule right hand with a number of symbols different from 2")
        if np.abs(np.sum(list(rules.values())) - 1) > 10 ** -10:
            print(np.sum(list(rules.values())))
            print("Found a law for derivations of root tag which does not sum to 1")

    for (word, tags) in my_CYK_parser.PCFG.lexicon.items():
        if np.abs(np.sum(list(tags.values())) - 1) > 10 ** -10:
            print(np.sum(list(tags.values())))
            print("Found a law for word tag which does not sum to 1")         
'''