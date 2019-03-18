from utils import *
from cyk import CYK_Parser

# Read arguments
p = argparse.ArgumentParser( description='Basic run script for the Parser' )

p.add_argument( '--inFile',    type=str, required=True,                       help='Input file (text to parse)' )
p.add_argument( '--outFile',   type=str, required=False, default = None,      help='Output file (will store parsings in bracketed format)' )
p.add_argument( '--vizTime',   type=bool, required=False, default=False,      help='Plot time of execution')
p.add_argument( '--vizOOV',   type=bool, required=False, default=False,       help='Plot management of OOV words')
p.add_argument( '--vizTree',  type=bool, required=False, default=False,       help='Plot the parsing as a Tree (requires pygraphviz library)' )

args = p.parse_args()

if args.vizTree: #requires pygraphviz library
   from draw_tree import draw_tree

#########################################################################################################
#BUILDING PARSER ON TRAIN CORPUS

file_corpus = open("data/SEQUOIA_treebank","r")
corpus = []
for line in file_corpus:
    corpus.append(line)
file_corpus.close()

frac_train = 0.8
N = len(corpus)
nb_train = int(round(N*frac_train))
corpus_train = corpus[:nb_train]

#Building Parser
print("Building CYK parser...")
tic = time.time()
my_CYK_parser = CYK_Parser(corpus_train)
tac = time.time()
if args.vizTime: print("Done in "+str(round(tac-tic,2))+"s")
else: print("Done")

#########################################################################################################
#PARSING INPUT DATA

print("Start Parsing Text")
print("")

for sent in open(args.inFile):

    print("#################")
    print("Sentence : ")
    print(sent)
    print("")

    print("Parsing")
    tic = time.time()
    my_parsing = my_CYK_parser.parse(sent, remove_artificial_symbols=True, viz_oov = args.vizOOV)   
    if my_parsing is None:
        print("Found no parsing grammatically valid.")
    else:
        print(my_parsing)
    tac = time.time()
    if args.vizTime: print("Done in " + str(round(tac - tic, 2)) + "s")
    print("")

    if args.vizTree:
        if not(my_parsing is None):
            draw_tree(my_parsing)

    if not(args.outFile is None):
        with open(args.outFile, 'a') as f:
            if my_parsing is None:
                f.write("Found no parsing grammatically valid." + "\n")
            else:
                f.write(my_parsing + "\n")
