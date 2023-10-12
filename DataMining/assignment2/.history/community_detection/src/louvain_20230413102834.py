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

def louvain_algorithm(G):
    # Initialize each node to be in its own community
    node_to_community = {node: node for node in G.nodes}
    
    # Initialize variables for modularity calculation
    m = G.number_of_edges()
    k = defaultdict(int)
    for node in G.nodes:
        k[node] = G.degree(node)
    Q = 0
    
    # Keep track of which communities have been merged
    merged_communities = set()
    
    # Loop until no further improvement is possible
    improved = True
    while improved:
        improved = False
        
        # Randomly shuffle nodes to avoid biases
        nodes = list(G.nodes)
        random.shuffle(nodes)
        
        # Iterate over nodes and try to move them to a better community
        for node in nodes:
            # Calculate the change in modularity for each possible move
            old_community = node_to_community[node]
            best_community = old_community
            delta_Q_max = 0
            for neighbor in G.neighbors(node):
                new_community = node_to_community[neighbor]
                if old_community != new_community:
                    # Calculate the change in modularity for this move
                    delta_Q = (G.has_edge(node, neighbor) - k[node]*k[neighbor]/(2*m)) \
                              + (k[node]/(2*m) - k[neighbor]/(2*m)) * sum(G.degree(x) for x in G.neighbors(node))
                    
                    # Update best move if this one is better
                    if delta_Q > delta_Q_max:
                        delta_Q_max = delta_Q
                        best_community = new_community
            
            # Move the node to the best community if it improves modularity
            if best_community != old_community:
                node_to_community[node] = best_community
                k[old_community] -= G.degree(node)
                k[best_community] += G.degree(node)
                Q += delta_Q_max
                improved = True
                merged_communities.add(old_community)
        
        # Merge communities that have been moved to the same new community
        community_mapping = {}
        new_community_index = 0
        for node in G.nodes:
            community = node_to_community[node]
            if community not in community_mapping:
                community_mapping[community] = new_community_index
                new_community_index += 1
            node_to_community[node] = community_mapping[community]
        
        # Rebuild the graph with the new communities
        communities = defaultdict(list)
        for node in G.nodes:
            community = node_to_community[node]
            communities[community].append(node)
        G = nx.DiGraph()
        for community in communities.values():
            G.add_node(community[0])
        for u, v in G.edges:
            u_community = node_to_community[u]
            v_community = node_to_community[v]
            if u_community != v_community:
                G.add_edge(u_community, v_community)
        
        # Stop iterating if no communities were merged on this iteration
        if len(merged_communities) == 0:
            break
    
    # Update the category attribute for each node based on its final community
    for node, community in node_to_community.items():
        G._node[node].update({'category':community})
    
    return G

### end of TODO ###


def main():
    G = getGraph()
    
    ### TODO ###
    # implement your community detection alg. here



    ### end of TODO ###

    store_result(G)

if __name__ == "__main__":
    main()