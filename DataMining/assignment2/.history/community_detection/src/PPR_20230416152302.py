# please use PPR algorithm to finish the community detection task
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
def storeResult(G):
    with open('../data/predictions_PPR.csv', 'w') as output:
        output.write("id,category\n")
        for i in range(NUM_NODES):
            output.write("{},{}\n".format(i, G._node[i]['category']))

# approximate PPR using push operation
def approximatePPR(G):


    return ppr

sweepCut
### TODO ###
### you can define some useful function here if you want
def push(G, u, r, q, beta):
    r_new = r
    q_new = q
    q_new[u] = 0.5 * beta * q[u]
    for v in G.neighbors(u):
        q_new[v] = q[v] + 0.5 * beta * (q[u] / G.degree(u))
    return r_new, q_new
### end of TODO ###


def main():
    G = getGraph()

    ### TODO ###
    # implement your community detection alg. here



    ### end of TODO ###

    storeResult(G)

if __name__ == "__main__":
    main()