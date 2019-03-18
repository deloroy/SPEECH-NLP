from imports import *

####################################################################################################
def sentence(tagged_sent):
    #return the sentence from a tagged_sentged sentence
    tagged_sent_splitted = tagged_sent.split() #into a list
    sent = []
    for bloc in tagged_sent_splitted:
        if (bloc[0]=="("):
            continue
        else:
            word = ""
            for caract in bloc:
                if (caract==")"):
                    break
                word += caract
            sent.append(word)
    return ' '.join(sent)


def non_functional_tag(functional_tag):
    #keep only the part of the tag before the hyphen
    tag = ""
    for caract in functional_tag:
        if caract=="-":
            break
        tag+=caract
    return tag


def all_symbols(grammar):
    #return the list of all symbols encountered in a grammar
    res =  []
    for (root_tag,rules) in grammar.items():
        res.append(root_tag)
        for list_tags in rules.keys():
            for tag in list_tags:
                res.append(tag)
    return list(np.unique(res))


####################################################################################################
def add(dico, word, tag, counts = 1):
    # incrementing dico[word][tag] by counts,
    # word is a string,
    # tag is a string or a list (will be converted to a tuple in such case)

    if type(tag) == list:
        tag = tuple(tag)
    if word in dico.keys():
        if tag in dico[word].keys():
            dico[word][tag] += counts
        else:
            dico[word][tag] = counts
    else:
        dico[word] = {tag: counts}

def normalize_counts(dico):
    #convert counts to probabilities
    #ex: perform for each idx, the transformation below :
    #dico[idx] = {i:c,j:d} ->  dico[idx] = {i:c/(c+d),j:d/(c+d)}

    res = deepcopy(dico)
    for (word,tags_counts) in dico.items():
        total_counts = np.sum(list(tags_counts.values()))
        for tag in tags_counts.keys():
            res[word][tag] /= total_counts
    return res


####################################################################################################
def postagged_sent_to_tree(tagged_sent, remove_after_hyphen=True):
    #return the parsing tree from a parsed sentence as a string in input
    max_id_node = 0

    g = nx.DiGraph()

    sent = tagged_sent.split()  # into a list
    hierarchy = []  # index = number of opened brackets since the beginning of the sentence
    # hierarchy[index] = list of tags pointed by root tag hierarchy[index-1]
    hierarchy.append([])  # list for level 0

    level = 0  # current difference between the number of opened brackets (minus the first one) and the number of closed brackets

    for (idx_bloc, bloc) in enumerate(sent):

        if (bloc[0] == "("):  # then the bloc is introducing a new tag

            if remove_after_hyphen:
                tag = non_functional_tag(bloc[1:])  # we add it to the hierarchy
            else:
                tag = bloc[1:]
            if level < len(hierarchy):  # there is already one tag as its level
                hierarchy[level].append((tag, max_id_node))
            else:  # first tag as its level
                hierarchy.append([(tag, max_id_node)])
            if idx_bloc > 0:
                g.add_node(max_id_node, name=tag)
                max_id_node += 1
            # print(hierarchy)
            level += 1  # since we opened a new bracket

        else:  # then the bloc is introducing the word name and the number of closing brackets

            word = ""
            nb_closing_brackets = 0
            for caract in bloc:
                if (caract == ")"):
                    nb_closing_brackets += 1
                else:
                    word += caract

            g.add_node(max_id_node, name=word)
            g.add_edge(max_id_node - 1, max_id_node)
            max_id_node += 1

            level -= nb_closing_brackets  # since we closed a bracket

            for k in range(nb_closing_brackets - 1, 0, -1):  # at least 2 brackets closed -> new grammar rule defined
                root = hierarchy[-2][-1][0]  # root tag
                id_root = hierarchy[-2][-1][1]
                if root == '':  # if the root is the beginning of the sentence
                    break
                tags = hierarchy[-1]  # child tags

                for tag in tags:
                    g.add_edge(id_root, tag[1])

                hierarchy.pop()  # popping from the hierarchy the childs list

    return g

def tree_to_postagged_sent_rec(T, node):
    # return the parsed sentence as a string from the parsing tree in input and rooted in node
    children = list(T.successors(node))
    if (len(children) == 1) and (len(list(T.successors(children[0]))) == 0):
        return "(" + T.nodes[node]["name"] + " " + T.nodes[children[0]]["name"] + ")"
    else:
        res = "(" + T.nodes[node]["name"]
        for child in sorted(children):
            res += " " + tree_to_postagged_sent_rec(T, child)
        res += ")"
        return res

def tree_to_postagged_sent(T):
    # return the parsed sentence as a string from the parsing tree in input
    root = list(nx.topological_sort(T))[0]
    return "( " + tree_to_postagged_sent_rec(T, root) + ")"



####################################################################################################
#I imported the three functions below from https://nbviewer.jupyter.org/gist/aboSamoor/6046170

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

DIGITS = re.compile("[0-9]", re.UNICODE)

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
