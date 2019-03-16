from utils import *

def postagged_sent_to_tree(postagged_sent, remove_after_hyphen=True):
    max_id_node = 0

    g = nx.DiGraph()

    sent = postagged_sent.split()  # into a list
    hierarchy = []  # index = number of opened brackets since the beginning of the sentence
    # hierarchy[index] = list of tags pointed by root tag hierarchy[index-1]
    hierarchy.append([])  # list for level 0

    level = 0  # current difference between the number of opened brackets (minus the first one) and the number of closed brackets

    for (idx_bloc, bloc) in enumerate(sent):

        if (bloc[0] == "("):  # then the bloc is introducing a new postag

            if remove_after_hyphen:
                postag = non_functional_tag(bloc[1:])  # we add it to the hierarchy
            else:
                postag = bloc[1:]
            if level < len(hierarchy):  # there is already one postag as its level
                hierarchy[level].append((postag, max_id_node))
            else:  # first postag as its level
                hierarchy.append([(postag, max_id_node)])
            if idx_bloc > 0:
                g.add_node(max_id_node, name=postag)
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
    root = list(nx.topological_sort(T))[0]
    return "( " + tree_to_postagged_sent_rec(T, root) + ")"

def draw_tree(postagged_sent):
    g = postagged_sent_to_tree(postagged_sent)
    
    plt.figure(figsize=(12,12))
    nx.draw(g, labels = nx.get_node_attributes(g, "name"), arrows=False, pos=graphviz_layout(g, prog='dot'))
    plt.show()