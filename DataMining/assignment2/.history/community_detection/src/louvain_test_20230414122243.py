# please use Louvain algorithm to finish the community detection task
# Do not change the code outside the TODO part
# Attention: after you get the category of each node, please use G._node[id].update({'category':category}) to update the node attribute
# you can try different random seeds to get the best result

import networkx as nx
import csv
import random
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

def main():
    G = getGraph()

    add_edge_weights(G)

    # run the Louvain algorithm
    communities = louvain(G)

    # assign community IDs to the communities
    communities = assign_community_ids(communities)

    # update the node attributes in the graph
    for id, category in communities.items():
        G._node[id].update({'category':category})

    # save the predictions to a CSV file
    store_result(G)

    # compute the modularity score for the detected communities
    modularity_score = compute_modularity(G, communities)

    print("Modularity score: {:.3f}".format(modularity_score))
if __name__ == "__main__":
    main()