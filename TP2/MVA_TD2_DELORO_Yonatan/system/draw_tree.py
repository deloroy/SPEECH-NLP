from utils import *
from networkx.drawing.nx_agraph import graphviz_layout

def draw_tree(postagged_sent):
    #draw the parsing tree
    #input : parsing as a string with nested brackets
    g = postagged_sent_to_tree(postagged_sent)
    
    plt.figure(figsize=(12,12))
    nx.draw(g, labels = nx.get_node_attributes(g, "name"), arrows=False, pos=graphviz_layout(g, prog='dot'))
    plt.show()
