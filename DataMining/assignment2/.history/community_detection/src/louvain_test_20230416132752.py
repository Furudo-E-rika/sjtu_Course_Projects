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
    return communities


def second_stage(G, communities):
    # initialize a dictionary to map nodes to their corresponding supernodes
    node_to_supernode = {node: com for node, com in communities.items()}
    
    # initialize a flag to indicate if the community configuration has changed
    changed = True
    
    while changed:
        # create a mapping from supernodes to their corresponding community partitions
        supernode_to_partition = defaultdict(list)
        for node, supernode in node_to_supernode.items():
            supernode_to_partition[supernode].append(node)
        
        # create a new graph of supernodes
        supernode_graph = nx.DiGraph()
        for supernode in supernode_to_partition:
            supernode_graph.add_node(supernode)
        
        # add weighted edges between supernodes based on the connections between their corresponding partitions
        for u, v, weight in G.edges(data='weight'):
            u_supernode = node_to_supernode[u]
            v_supernode = node_to_supernode[v]
            if u_supernode != v_supernode:
                edge_weight = supernode_graph.get_edge_data(u_supernode, v_supernode, default={'weight':0})['weight']
                supernode_graph.add_edge(u_supernode, v_supernode, weight=edge_weight+weight)

    
    return supernode_graph, supernode_to_partition

def get_neighbors_communities(G, node, communities):
    neighbors = set(G.neighbors(node))
    comms = set()
    for neighbor in neighbors:
        comms.add(communities[neighbor])
    return comms


def get_modularity_gain(G, node, com_neighbor, communities):
    k_i_in = 0
    k_i_tot = 0
    w_i_in = 0
    for neighbor in G.neighbors(node):
        if communities[neighbor] == com_neighbor:
            k_i_in += G[neighbor][node]['weight']
            w_i_in += G[neighbor][node]['weight']
        k_i_tot += G[neighbor][node]['weight']
    k_i_out = k_i_tot - k_i_in
    w_i_out = G.degree(node, weight='weight') - w_i_in
    sigma_tot = G.size(weight='weight')
    sigma_in = G.degree(node, weight='weight')
    sigma_c = 0
    w_c = 0
    for neighbor in G.neighbors(node):
        if communities[neighbor] == com_neighbor:
            sigma_c += G.degree(neighbor, weight='weight')
            w_c += G[neighbor][node]['weight']
    m_c = sigma_c / sigma_tot
    return (2 * w_i_in - k_i_tot * m_c) - (2 * w_i_out - k_i_tot * (1 - m_c)) - (2 * (sigma_in - w_i_in) - k_i_tot) * (2 * (sigma_tot - sigma_in - w_i_out) - k_i_tot) / sigma_tot

def main():
    G = getGraph()
    print("Number of nodes: ", G.number_of_nodes())
    print("Number of edges: ", G.number_of_edges())
    
    for e in enumerate(G.edges()):
        
        G.[e[0]][e[1]]['weight'] = 1

    communities = first_stage(G)
    print("done first phase")
    while True:
        contracted_G, new_partition = second_stage(G, communities)
        print("Number of nodes: ", contracted_G.number_of_nodes())
        print("Number of edges: ", contracted_G.number_of_edges())
        new_communities = first_stage(contracted_G)

        node_to_supernode = {}
        for supernode, supernode_community in new_communities.items():
            for node in new_partition[supernode]:
                node_to_supernode[node] = supernode_community

        if new_communities == communities and G == contracted_G:
            break
    
    for node, com in node_to_supernode.items():
        G._node[node].update({'category': com})
    store_result(G)
    

    
if __name__ == "__main__":
    main()