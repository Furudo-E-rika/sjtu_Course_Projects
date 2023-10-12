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
def _first_phase(G, m, partition):
    node2com = {u:i for i, u in enumerate(G.nodes())}
    inner_partition = {{u} for u in G.nodes()}

    in_degrees = dict(G.in_degree(weight="weight"))
    out_degrees = dict(G.out_degree(weight="weight"))
    Stot_in = [deg for deg in in_degrees.values()]
    Stot_out = [deg for deg in out_degrees.values()]

    neighbors = {u:{v: data["weight"] for v, data in G[u].items() if v!= u} for u in G}
    converge = False

    while not converge:
        for u in list(G.nodes):
            converge = True
            best_modularity = 0
            best_community = node2com[u]
            weight2com = _neighbor_weights(neighbors[u], node2com)
            in_degree = in_degrees[u]
            out_degree = out_degrees[u]
            Stot_in[best_community] -= in_degree
            Stot_out[best_community] -= out_degree

            for neigh_comm, weight in weight2com.items():
                gain = (weight - 1) * (out_degree * Stot_in[neigh_comm] + in_degree * Stot_out[neigh_comm])
                if gain > best_modularity:
                    best_modularity = gain
                    best_community = neigh_comm
            
            Stot_in[best_community] += in_degree
            Stot_out[best_community] += out_degree

            if best_community != node2com[u]:
                converge = False
                comm = G.nodes[u].get("nodes", {u})

def _neighbor_weights(neigh, node2com):
    weights = defa

def main():
    G = getGraph()


    print("Modularity score: {:.3f}".format(modularity_score))
if __name__ == "__main__":
    main()