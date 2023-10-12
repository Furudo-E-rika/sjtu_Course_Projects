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
                    q += (1 - (G.degree(i) * G.degree(j)) / (2 * m))

    return q / (2 * m)

def partitioning(G, partition):

    not_converge = True
    m = sum(weight for _, _, weight in G.edges.data('weight'))
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
                
                in_weight = sum([G[node][i]['weight'] for i in G.neighbors(node) if partition[i] == curr_community])
                tot_weight = sum([G[node][i]['weight'] for i in G.neighbors(node)])
                ki = sum([G[i][node]['weight'] for i in G.neighbors(node)])
                ki_in = sum([G[node][i]['weight'] for i in G.neighbors(node) if partition[i] == curr_community])
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
    
    for node1, node2, weight in G.edges.data('weight'):
        comm1 = partition[node1]
        comm2 = partition[node2]

        if contracted_G.has_edge(comm1, comm2):
            contracted_G[comm1][comm2]['weight'] += weight
        else:
            contracted_G.add_edge(comm1, comm2, weight=weight)

    while True:

        new_partition = dict()
        nodes = set(contracted_G.nodes())
        if len(nodes) == 1:
            break
            
        ## picking two communities that have the maximum edge weight between them
        max_weight = 0
        for comm1, comm2, data in contracted_G.edges(data=True):
            if data['weight'] > max_weight:
                max_weight = data['weight']
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
                if contracted_G.has_edge(max_comm1, comm):
                    new_weight_out += contracted_G[new_comm][comm]['weight']
                if contracted_G.has_edge(max_comm2, comm):
                    new_weight_out += contracted_G[new_comm][comm]['weight']
                if contracted_G.has_edge(comm, max_comm1):
                    new_weight_in += contracted_G[comm][new_comm]['weight']
                if contracted_G.has_edge(comm, max_comm2):
                    new_weight += contracted_G[comm][new_comm]['weight']
                contracted_G.add_edge(new_comm, comm, weight=new_weight)
        
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



    ### end of TODO ###

    store_result(G)

if __name__ == "__main__":
    main()