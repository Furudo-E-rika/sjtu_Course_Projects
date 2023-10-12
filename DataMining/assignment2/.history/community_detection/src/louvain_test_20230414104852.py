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

def modularity(G, communities, m):
    modularity = 0.0
    for community in set(communities.values()):
        # get the subgraph induced by nodes in the community
        nodes = [node for node in communities if communities[node] == community]
        subgraph = G.subgraph(nodes)

        # compute the degree sum and edge count for the community
        degree_sum = sum(subgraph.degree(node, weight='weight') for node in nodes)
        edge_count = sum(subgraph.degree(node, weight='weight') for node in subgraph.predecessors(node))

        # compute the modularity contribution for the community
        modularity += edge_count / (2 * m) - ((degree_sum / (2 * m)) ** 2)

    return modularity

def compute_modularity(G, communities):
    # compute the total number of edges in the graph
    m = sum(weight for _, _, weight in G.edges.data('weight'))

    # compute the modularity score
    modularity_score = modularity(G, communities, m)

    return modularity_score

def initialize_communities(G):
    communities = {n: i for i, n in enumerate(G.nodes())}
    return communities

def move_node(G, node, communities):
    # find the best community for the node
    best_community, best_increase = find_best_community(G, node, communities)
    
    # move the node to the best community
    communities[node] = best_community
    
    # update the modularity score
    modularity_score = modularity(G, communities)
    
    return modularity_score, communities

def assign_community_ids(communities):
    community_ids = {}
    current_id = 0
    
    for node, community in communities.items():
        if community not in community_ids:
            community_ids[community] = current_id
            current_id += 1
    
    new_communities = {}
    
    for node, community in communities.items():
        new_communities[node] = community_ids[community]
    
    return new_communities

def compute_delta_modularity(G, node, communities, community, degree_sum, edge_count, modularity_score):
    # compute the change in the degree sum and edge count for the node's current community
    current_degree_sum = sum(G.degree(node, weight='weight') for node in communities if communities[node] == communities[node])
    current_edge_count = sum(G.degree(node, weight='weight') for node in G.predecessors(node) if communities[node] == community)
    
    # compute the change in the degree sum and edge count for the target community
    target_degree_sum = degree_sum[community] + G.degree(node, weight='weight')
    target_edge_count = edge_count[community] + current_edge_count
    
    # compute the change in modularity score
    delta_modularity = target_edge_count / (2 * m) - ((target_degree_sum / (2 * m)) ** 2) - (current_edge_count / (2 * m)) + ((current_degree_sum / (2 * m)) ** 2)
    
    return delta_modularity

def find_best_community(G, node, communities):
    # get the current community of the node
    current_community = communities[node]
    
    # find the community that yields the highest increase in modularity if the node is moved there
    best_community = current_community
    best_increase = 0.0
    
    neighbors = list(G.neighbors(node))
    random.shuffle(neighbors)
    
    for neighbor in neighbors:
        if neighbor == node:
            continue
        
        # get the current community of the neighbor
        neighbor_community = communities[neighbor]
        
        # compute the increase in modularity if the node is moved to the neighbor's community
        delta_modularity = compute_delta_modularity(G, node, current_community, neighbor_community)
        
        if delta_modularity > best_increase:
            best_community = neighbor_community
            best_increase = delta_modularity
    
    return best_community, best_increase

def louvain(G):
    # initialize each node to be in its own community
    communities = {n: i for i, n in enumerate(G.nodes())}
    
    # keep track of the modularity at each iteration
    modularity = compute_modularity(G, communities)
    max_modularity = modularity
    
    while True:
        # keep track of whether any moves have been made during this iteration
        moved = False
        
        # iterate over the nodes in a random order
        nodes = list(G.nodes())
        random.shuffle(nodes)
        
        # find the best community for each node
        for node in nodes:
            # get the current community of the node
            current_community = communities[node]
            
            # find the community that yields the highest increase in modularity if the node is moved there
            best_community = current_community
            best_increase = 0.0
            
            neighbors = list(G.neighbors(node))
            random.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor == node:
                    continue
                
                # get the current community of the neighbor
                neighbor_community = communities[neighbor]
                
                # compute the increase in modularity if the node is moved to the neighbor's community
                delta_modularity = compute_delta_modularity(G, node, current_community, neighbor_community)
                
                if delta_modularity > best_increase:
                    best_community = neighbor_community
                    best_increase = delta_modularity
            
            # move the node to the best community, if it improves modularity
            if best_community != current_community:
                communities[node] = best_community
                modularity += best_increase
                moved = True
        
        # stop if no moves were made during this iteration
        if not moved:
            break
        
        # update the modularity and the maximum modularity
        if modularity > max_modularity:
            max_modularity = modularity
    
    # assign a unique ID to each community
    community_id = 0
    community_ids = {}
    
    for node, community in communities.items():
        if community not in community_ids:
            community_ids[community] = community_id
            community_id += 1
        
        G.nodes[node]['category'] = community_ids[community]


def main():
    G = getGraph()
    
    ### TODO ###
    # implement your community detection alg. here
    
    ### end of TODO ###

    store_result(G)

if __name__ == "__main__":
    main()