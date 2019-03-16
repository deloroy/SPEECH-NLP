from utils import *
from cyk import CYK_Parser

file_corpus = open(path_to_data+"SEQUOIA_treebank","r")
corpus = []
for line in file_corpus:
    corpus.append(line)
file_corpus.close()
#print(corpus[0:10])


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


#Building Parser
Parser = CYK_Parser(corpus_train)



print(len(Parser.PCFG.list_all_symbols[:Parser.PCFG.nb_tags]), " grammatical tags")
print(len(Parser.PCFG.list_all_symbols[Parser.PCFG.nb_tags:]), " symbols (with artificial ones)")

print("")

print(len(Parser.Tagger.words_lexicon), " words in lexicon")
print(len(Parser.Tagger.Meaning_Search_Engine.id_word_lexicon), " words in lexicon having an embedding")
print(len(Parser.Tagger.words_with_embeddings), " words having an embedding")
#print(Parser.Tagger.Meaning_Search_Engine.embeddings_lexicon.shape)



if False: #test oov print("test of the oov tagger")
    print("")

    sentences_test = [sentence(postag) for postag in corpus_test]
    print(sentences_test[0])
    print("")

    for word in sentences_test[0].split():
        print(word, Parser.Tagger.tag(word,viz_oov = True))
        print("")
    print("")

    print("##################################")
    print("")


if False:  # test cyk
    Parser = CYK_Parser(corpus_train)
    Parser.PCFG.grammar = {"SENT": {("GRP1", "GRP2"): 1}, "GRP1": {("ADV", "VERB"): 1},
                                "GRP2": {("ART", "NOM"): 1}}
    tagger = {"Il": {"ADV": 1}, "demande": {"VERB": 1}, "le": {"ART": 1}, "renvoi": {"NOM": 1}}
    list_all_symbols = ["SENT", "GRP1", "GRP2", "ADV", "VERB", "ART", "NOM", "PUNCT"]  # redefining variable
    Parser.PCFG.nb_tags = 8
    Parser.PCFG.nb_all_symbols = len(list_all_symbols)  # redefining variable
    Parser.symbol_to_id = {tag: i for (i, tag) in enumerate(list_all_symbols)}

else:
    for (root, rules) in Parser.PCFG.grammar.items():
        for (right, proba) in rules.items():
            # print(root, right, proba)
            if len(right) != 2:
                print("Found a rule right hand with a number of symbols different from 2")
        if np.abs(np.sum(list(rules.values())) - 1) > 10 ** -10:
            print(np.sum(list(rules.values())))
            print("Found a law for derivations of root tag which does not sum to 1")

    for (word, tags) in Parser.PCFG.lexicon.items():
        if np.abs(np.sum(list(tags.values())) - 1) > 10 ** -10:
            print(np.sum(list(tags.values())))
            print("Found a law for word tag which does not sum to 1")

postagged_sent = corpus_test[1]

print("Sentence")
#sent = sentence(postagged_sent)
sent = "La cours a demandé la cours ."
print(sent)
print("")

#print("Ground truth : ")
#print(postagged_sent)
#print("")

bools = [True] #, False
for bool in bools:
    print("My parsing : ")
    parsing = Parser.parse(sent, remove_artificial_symbols=bool, viz_oov = True)
    print(parsing)

    from utils_draw_tree import draw_tree
    draw_tree(parsing)