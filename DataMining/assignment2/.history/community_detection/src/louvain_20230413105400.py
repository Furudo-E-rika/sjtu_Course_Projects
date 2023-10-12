# please use Louvain algorithm to finish the community detection task
# Do not change the code outside the TODO part
# Attention: after you get the category of each node, please use G._node[id].update({'category':category}) to update the node attribute
# you can try different random seeds to get the best result

import networkx as nx
import csv
# you can use basic operations in networkx
# you can also import other libraries if you need, but do not use any community detection APIs

NUM_NODES = 31136

# read edges.csv and construct the graph
def getGraph():
    G = nx.DiGraph()
    for i in range(NUM_NODES):
        G.add_node(i)
    with open("../data/lab1_edges.csv", 'r') as csvFile:
        csvFile.readline()
        csv_reader = csv.reader(csvFile)
        for row in csv_reader:
            source = int(row[0])
            target = int(row[1])
            G.add_edge(source, target)
    print("graph ready")
    return G

# save the predictions to csv file
# Attention: after you get the category of each node, please use G._node[id].update({'category':category}) to update the node attribute
def store_result(G):
    with open('../data/predictions_louvain.csv', 'w') as output:
        output.write("id,category\n")
        for i in range(NUM_NODES):
            output.write("{},{}\n".format(i, G._node[i]['category']))


### TODO ###
### you can define some useful function here if you want
from collections import defaultdict
import random

def modularity(G, partition):
    m = G.number_of_edges()
    q = 0
    for c in partition.values():
        c_nodes = set(c)
        for i in c_nodes:
            for j in c_nodes:
                if G.has_edge(i, j):
                    q += (1 - (G.degree(i) * G.degree(j)) / (2))
        for i in c_nodes:
            q -= G.degree(i) * len(c_nodes) / (2 * m)
    return q / (2 * m)

### end of TODO ###


def main():
    G = getGraph()
    
    ### TODO ###
    # implement your community detection alg. here



    ### end of TODO ###

    store_result(G)

if __name__ == "__main__":
    main()