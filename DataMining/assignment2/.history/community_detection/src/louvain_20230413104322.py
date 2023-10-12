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

def partition(G):
    # initialize each node to be in its own community
    partition = {node: node for node in G.nodes()}
    modularity = -1  # initialize modularity

    # repeat until modularity no longer increases
    while True:
        # get the neighboring communities for each node
        neighbor_communities = {node: set([partition[neighbor] for neighbor in G.neighbors(node)]) for node in G.nodes()}

        # get the modularity contribution of each node moving to each neighboring community
        delta_Q = {}
        for node in G.nodes():
            for neighbor_community in neighbor_communities[node]:
                # modularity change from moving node to its neighbor's community
                delta = (nx.degree(G, node, weight='weight') * (nx.degree(G, node, weight='weight') + sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))) - 2 * nx.degree(G, node, weight='weight') * sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node)) / (2 * G.size())) / (2 * G.size())
                delta_Q[(node, neighbor_community)] = delta

        # get the (node, community) pair that maximizes modularity
        node_community, delta_Q_value = max(delta_Q.items(), key=lambda x: x[1])

        # if the modularity no longer increases, return the final partition
        if delta_Q_value <= 0:
            break

        # otherwise, move the node to its neighboring community with highest modularity increase
        partition[node_community[0]] = node_community[1]
        modularity += delta_Q_value

    return partition


### end of TODO ###


def main():
    G = getGraph()
    
    ### TODO ###
    # implement your community detection alg. here



    ### end of TODO ###

    store_result(G)

if __name__ == "__main__":
    main()