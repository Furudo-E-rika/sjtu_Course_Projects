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


### TODO ###
### you can define some useful function here if you want
from collections import defaultdict

def compute_modularity(G, partitions, m):
    score = 0
    for partition in partitions:
        
def node2comm(G):
    _node2comm = {}
    edge_weights = defaultdict(lambda: defaultdict(float))
    for i, node in enumerate(G.nodes()):
        _node2comm[node] = i  ## set each node itself as a community
        for edge in G[node].items():
            edge_weights[node][edge[0]] = edge[1].get('weight', 1)
    return _node2comm, edge_weights

def partitioning(G, partition):
    
    not_converge = False
    m = sum(G[u][v].get('weight', 1) for u, v in G.edges)
    
    while not_converge:

        nodes = list(G.nodes())
        random.shuffle(nodes)
        
        for node in nodes:
            
            curr_community = partition[node]
            mod_gains = defaultdict()
            for neighbor in G.neighbors(node):
                
                neighbor_community = partition[neighbor]
                if neighbor_community == curr_community:
                    continue
                
                in_weight = sum([G[node][i].get('weight', 1) for i in G.neighbors(node) if partition[i] == curr_community])
                tot_weight = sum([G[node][i].get('weight', 1) for i in G.neighbors(node)])
                ki = sum([G[i][node].get('weight', 1) for i in G.neighbors(node)])
                ki_in = sum([G[node][i].get('weight', 1) for i in G.neighbors(node) if partition[i] == curr_community])
                delta_Q_add = ((in_weight + ki_in) / (2*m) - ((tot_weight + ki)/(2*m)) ** 2) - (in_weight/(2*m) - (tot_weight/(2*m))**2 - (ki/(2*m))**2)
                delta_Q_remove = ((in_weight - ki_in) / (2*m) - ((tot_weight - ki)/(2*m)) ** 2) - (in_weight/(2*m) - (tot_weight/(2*m))**2 - (ki/(2*m))**2)
            
                mod_gains[neighbor_community] = delta_Q_add + delta_Q_remove
                    
            new_community = max(mod_gains, key=mod_gains.get)
           
            if mod_gains[new_community] > 0:
                partition[node] = new_community
                not_converge = True

    return partition


def restructuring(G, partition):
    contracted_G = nx.DiGraph()

    communities = set(partition.values())
    for comm in communities:
        contracted_G.add_node(comm)
    
    for node1, node2 in G.edges:
        
        comm1 = partition[node1]
        comm2 = partition[node2]

        if contracted_G.has_edge(comm1, comm2):
            contracted_G[comm1][comm2]['weight'] = contracted_G[comm1][comm2].get('weight', 1) + G[node1][node2].get('weight', 1)
        else:
            
            contracted_G.add_edge(comm1, comm2, weight=G[node1][node2].get('weight', 1))

    while True:

        new_partition = dict()
        nodes = set(contracted_G.nodes())
        if len(nodes) == 1:
            break
            
        ## picking two communities that have the maximum edge weight between them
        max_weight = 0
        for comm1, comm2, data in contracted_G.edges(data=True):
            if data['weight'] > max_weight:
                max_weight = data.get('weight', 1)
                max_comm1 = comm1
                max_comm2 = comm2
        
        ## merge these two communities into a new supernode
        new_comm = max(nodes) + 1
        contracted_G.add_node(new_comm)
        for node, comm in partition.items():
            if comm == max_comm1 or comm == max_comm2:
                new_partition[node] = new_comm
        for comm in nodes:
            if comm != max_comm1 and comm != max_comm2:
                new_weight_out = 0
                new_weight_in = 0
                if contracted_G.has_edge(max_comm1, comm):
                    new_weight_out += contracted_G[max_comm1][comm].get('weight', 1)
                if contracted_G.has_edge(max_comm2, comm):
                    new_weight_out += contracted_G[max_comm2][comm].get('weight', 1)
                if contracted_G.has_edge(comm, max_comm1):
                    new_weight_in += contracted_G[comm][max_comm1].get('weight', 1)
                if contracted_G.has_edge(comm, max_comm2):
                    new_weight_in += contracted_G[comm][max_comm2].get('weight', 1)
                contracted_G.add_edge(new_comm, comm, weight=new_weight_out)
                contracted_G.add_edge(comm, new_comm, weight=new_weight_in)
        
        # Update the partition and check if it has stabilized
        if new_partition == partition:
            break
        partition = new_partition

    return contracted_G, partition

### end of TODO ###


def main():
    G = getGraph()
    
    ### TODO ###
    # implement your community detection alg. here
    partition = {node: node for node in G.nodes()}
    while True:
        old_modularity = compute_modularity(G, partition)
        partition = partitioning(G, partition)
        contracted_G, new_partition = restructuring(G, partition)
        new_modularity = compute_modularity(contracted_G, new_partition)
        print(old_modularity, new_modularity)
        if(abs(new_partition - old_modularity) < 0.1):
            break


    for node in G.nodes():
        G.nodes[node]['category'] = partition[node]
    ### end of TODO ###

    store_result(G)

if __name__ == "__main__":
    main()