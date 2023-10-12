# please use Louvain algorithm to finish the community detection task
# Do not change the code outside the TODO part
# Attention: after you get the category of each node, please use G._node[id].update({'category':category}) to update the node attribute
# you can try different random seeds to get the best result

import networkx as nx
import csv
from collections import defaultdict

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


def first_stage(G):
    communities = defaultdict(lambda: len(communities))
    for i in G.nodes:
        communities[i] = i
    improvement = True
    while improvement:
        improvement = False
        for node in G.nodes:
            com_node = communities[node]
            best_com = com_node
            best_gain = 0
            for com_neighbor in get_neighbors_communities(G, node, communities):
                gain = get_modularity_gain(G, node, com_neighbor, communities)
                if gain > best_gain:
                    best_com = com_neighbor
                    best_gain = gain
            if best_com != com_node:
                communities[node] = best_com
                improvement = True
    return communities, improvement


def second_stage(G, communities):
    reverse_communities = defaultdict(list)
    for node, com in communities.items():
        reverse_communities[com].append(node)
    mapping = {}
    for com, nodes in reverse_communities.items():
        subgraph = G.subgraph(nodes)
        subcommunities = first_stage(subgraph)
        for node, subcom in subcommunities.items():
            mapping[node] = communities[nodes[subcom]]
    return mapping


def get_neighbors_communities(G, node, communities):
    neighbors = set(G.neighbors(node))
    comms = set()
    for neighbor in neighbors:
        comms.add(communities[neighbor])
    return comms


def get_modularity_gain(G, node, com_neighbor, communities):
    k_i_in = 0
    k_i_tot = 0
    for neighbor in G.neighbors(node):
        if communities[neighbor] == com_neighbor:
            k_i_in += 1
        k_i_tot += 1
    k_i_out = k_i_tot - k_i_in
    sigma_tot = G.number_of_edges()
    sigma_in = G.in_degree(node)
    sigma_c = 0
    for neighbor in G.neighbors(node):
        if communities[neighbor] == com_neighbor:
            sigma_c += G.in_degree(neighbor)
    m_c = sigma_c / (2 * sigma_tot)
    return (2 * k_i_in - k_i_tot * m_c) - (2 * sigma_in - k_i_tot) * (2 * m_c)
    
    

def main():
    G = getGraph()
    print("Number of nodes: ", G.number_of_nodes())
    print("Number of edges: ", G.number_of_edges())
    communities = first_stage(G)
    mapping = second_stage(G, communities)
    for node, com in mapping.items():
        G._node[node].update({'category': com})
    store_result(G)
    

    
if __name__ == "__main__":
    main()