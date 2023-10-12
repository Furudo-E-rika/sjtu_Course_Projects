import networkx as nx
import csv
import random

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
def storeResult(G):
    with open('../data/predictions_PPR.csv', 'w') as output:
        output.write("id,category\n")
        for i in range(NUM_NODES):
            output.write("{},{}\n".format(i, G._node[i]['category']))

# approximate PPR using push operation
def approximatePPR(G, alpha=0.1, num_iter=100):
    ppr = {}
    for node in G.nodes():
        ppr[node] = 0
    start = random.choice(list(G.nodes()))
    ppr[start] = 1
    for i in range(num_iter):
        for node in G.nodes():
            if ppr[node] > 0:
                out_neighbors = list(G.successors(node))
                if out_neighbors:
                    weight = alpha * ppr[node] / len(out_neighbors)
                    for neighbor in out_neighbors:
                        ppr[neighbor] += weight
        ppr[start] += (1 - alpha) * ppr[start]
    return ppr

# compute conductance of a set of nodes
def computeConductance(G, nodes):
    cut_size = 0
    vol_A = 0
    for node in nodes:
        vol_A += G.degree(node)
        for neighbor in G.successors(node):
            if neighbor not in nodes:
                cut_size += 1
    if vol_A == 0:
        return 1
    else:
        return cut_size / min(vol_A, G.number_of_edges() - vol_A)

# label nodes using sweep cut algorithm
def sweepCut(G, ppr):
    nodes = list(G.nodes())
    random.shuffle(nodes)
    best_cond = 1
    best_set = set()
    for i in range(len(nodes)):
        node = nodes[i]
        if node in best_set:
            continue
        set1 = set(nodes[:i+1])
        set2 = set(nodes[i+1:])
        cond1 = computeConductance(G, set1)
        cond2 = computeConductance(G, set2)
        if min(cond1, cond2) < best_cond:
            best_cond = min(cond1, cond2)
            best_set = set1 if cond1 < cond2 else set2
    for node in G.nodes():
        G._node[node].update({'category': 1 if node in best_set else -1})

def main():
    G = getGraph()
    ppr = approximatePPR(G)
    sweepCut(G, ppr)
    storeResult(G)

if __name__ == "__main__":
    main()