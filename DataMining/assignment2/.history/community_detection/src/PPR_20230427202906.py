# please use PPR algorithm to finish the community detection task
# Do not change the code outside the TODO part
# Attention: after you get the category of each node, please use G._node[id].update({'category':category}) to update the node attribute
# you can try different random seeds to get the best result

import networkx as nx
import csv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
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
def storeResult(G):
    with open('../data/predictions_PPR.csv', 'w') as output:
        output.write("id,category\n")
        for i in range(NUM_NODES):
            output.write("{},{}\n".format(i, G._node[i]['category']))

# approximate PPR using push operation
def approximatePPR(G, s, beta, tau):
    n = len(G.nodes())
    r = np.zeros(n)
    q = np.zeros(n)
    q[s] = 1

    while np.max(q / np.array([G.degree(v) for v in G])) >= tau:
        u = np.argmax(q / np.array([G.degree(v) for v in G]))
        r, q = push(G, u, r, q, beta)
        
    return r


def conductance(G, S, m):
    cut_size = sum([1 for u, v in G.edges() if u in S and v not in S])
    volume = min(sum(G.out_degree(v) for v in S), 2 * m - sum(G.out_degree(v) for v in S))
    conductance = cut_size / volume
    return cut_size, volume, conductance

def conductance_gain(G, S, node, old_cut_size, old_volume, m):
    new_cut_size = old_cut_size
    new_volume = min(old_volume + G.out_degree(node), 2 * m - old_volume - G.out_degree(node))
    
    for _, v in G.out_edges(node):
        if v in S:
            new_cut_size -= 1
    
    for v, _ in G.in_edges(node):
        if v not in S:
            new_cut_size += 1
    
    _conductance = new_cut_size / new_volume

    return _conductance, new_cut_size, new_volume

# use PPR to compute conductance and label the nodes
def sweepCut(G, ppr):
    sorted_nodes = np.argsort(ppr)[::-1]
    S = []
    best_conductance = float('inf')
    best_set = []
    conductance_list = []
    
    cut_size = 0
    volume = 0
    m = G.number_of_edges()
    
    for node in tqdm(sorted_nodes):
        if len(S) == 0:
            S.append(node)
            cut_size, volume, _conductance = conductance(G, S, m)
            best_conductance = _conductance
            conductance_list.append(_conductance)
            continue

        S.append(node) 
        _conductance, cut_size, volume = conductance_gain(G, S, node, cut_size, volume, m)
    
        conductance_list.append(_conductance)

        if _conductance < best_conductance:
            best_conductance = _conductance
            best_set = S.copy()

        if len(S) == 1000:
            
    
    return conductance_list, best_set

### TODO ###
### you can define some useful function here if you want
def push(G, u, r, q, beta):
    r_new = r
    q_new = q
    r_new[u] += (1 - beta) * q[u]
    q_new[u] = 0.5 * beta * q[u]
    for v in G.neighbors(u):
        q_new[v] = q[v] + 0.5 * beta * (q[u] / G.degree(u))
    return r_new, q_new
### end of TODO ###


def main():
    G = getGraph()

    ### TODO ###
    # implement your community detection alg. here

    # Parameters for the Approximate PPR algorithm
    s = 0  # Start node
    beta = 0.85
    tau = 1e-5

    ppr = approximatePPR(G, s, beta, tau)
    conductance_list, best_set = sweepCut(G, ppr)

    for node in G.nodes():
        if node in best_set:
            G._node[node].update({'category':1})
        else:
            G._node[node].update({'category':0})

    plt.figure()
    plt.plot(conductance_list)
    plt.xlabel('Size')
    plt.ylabel('Conductance')
    plt.savefig('Conductance.png')


    ### end of TODO ###

    storeResult(G)

if __name__ == "__main__":
    main()