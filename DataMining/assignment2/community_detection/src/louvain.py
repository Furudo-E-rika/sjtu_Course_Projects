# please use Louvain algorithm to finish the community detection task
# Do not change the code outside the TODO part
# Attention: after you get the category of each node, please use G._node[id].update({'category':category}) to update the node attribute
# you can try different random seeds to get the best result

import networkx as nx
import csv
import random
from tqdm import tqdm
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
class Louvain():
    def __init__(self, G):
        self.G = G
        self.m = G.number_of_edges()
        self.epsilon = 0.01
        self.node2comm_list = []

    def init_node2comm(self):
        self.node2comm = {node:i for i, node in enumerate(self.G.nodes())}
        self.node2comm_list.append(self.node2comm)
        
    def init_comm2node(self):
        comm2node = defaultdict(list)
        for node, comm in self.node2comm.items():
            comm2node[comm].append(node)
        self.comm2node = comm2node
    
    def compute_m(self):
        self.m = sum([self.G[u][v].get('weight', 1) for u, v in self.G.edges()])

    def modularity(self):
    
        modu_value = 0
        out_degree = dict(self.G.out_degree(weight="weight"))
        in_degree = dict(self.G.in_degree(weight="weight"))
        m = sum(out_degree.values())
        
        for comm, nodes in self.comm2node.items():

            weight_within = sum(weight for _, v, weight in self.G.edges(nodes, data="weight", default=1) if v in nodes)
            out_degree_sum = sum(out_degree[u] for u in nodes)
            in_degree_sum = sum(in_degree[u] for u in nodes) 

            modu_value +=  weight_within / m - out_degree_sum * in_degree_sum / m**2

        return modu_value


    def modularity_gain(self, n, origin, new):
        m = self.m
        in_weight_origin = sum(self.G[u][n].get('weight', 1) for u in self.G.predecessors(n) if u in origin)
        in_weight_new = sum(self.G[u][n].get('weight', 1) for u in self.G.predecessors(n) if u in new)
        tot_weight_origin = sum(self.G.degree(u, 'weight') for u in origin)
        tot_weight_new = sum(self.G.degree(u, 'weight') for u in new)
        ki = self.G.degree(n, 'weight')
        
        delta_Q_add = ((in_weight_new + in_weight_origin) / (2 * m) - ((tot_weight_new + ki) / (2 * m)) ** 2) - (
            in_weight_new / (2 * m) - (tot_weight_new / (2 * m)) ** 2 - (ki / (2 * m)) ** 2
        )
        
        delta_Q_remove = ((in_weight_origin - in_weight_origin) / (2 * m) - ((tot_weight_origin - ki) / (2 * m)) ** 2) - (
            in_weight_origin / (2 * m) - (tot_weight_origin / (2 * m)) ** 2 - (ki / (2 * m)) ** 2
        )
        
        mod_gain = delta_Q_add + delta_Q_remove
        
        return mod_gain

    def partitioning(self):
        
        not_converge = True
        m = self.m
        
        while not_converge:
            not_converge = False
            nodes = list(self.G.nodes())
            random.shuffle(nodes)
            
            for node in tqdm(nodes):

                curr_comm = self.comm2node[self.node2comm[node]]
                max_mod_gain = 0
                best_comm = -1
                
                for neighbor in self.G.neighbors(node):
                    neighbor_comm = self.comm2node[self.node2comm[neighbor]]
                    if neighbor_comm == curr_comm:
                        continue
                    mod_gain = self.modularity_gain(node, curr_comm, neighbor_comm)
                    
                    if mod_gain > max_mod_gain:
                        max_mod_gain = mod_gain
                        best_comm = self.node2comm[neighbor]
                        not_converge = True

                if best_comm != -1:
                    self.comm2node[self.node2comm[node]].remove(node)
                    self.comm2node[best_comm].append(node)
                    self.node2comm[node] = best_comm
            


    def restructuring(self):
        new_G = nx.DiGraph()
        for comm in self.comm2node.keys():
            new_G.add_node(comm)

        for u in tqdm(self.G.nodes()):
            for v in self.G.nodes():
                if self.G.has_edge(u, v):
                    if new_G.has_edge(self.node2comm[u], self.node2comm[v]):
                        new_G[self.node2comm[u]][self.node2comm[v]]['weight'] += self.G[u][v].get('weight', 1)
                    else:
                        new_G.add_edge(self.node2comm[u], self.node2comm[v], weight=self.G[u][v].get('weight', 1))
                        
        print("origin_nodes:", len(self.G.nodes()))
        print("restruc_nodes:", len(new_G.nodes()))
        self.G = new_G

    def louvain(self):
        origin_modularity = -1
        while True:
            self.init_node2comm()
            self.init_comm2node()
            new_modularity = self.modularity()

            print('modularity gain=', new_modularity - origin_modularity)
            if new_modularity - origin_modularity > self.epsilon:
                origin_modularity = new_modularity
            else:
                break
            
            self.compute_m()
            self.partitioning()
            self.restructuring()
            
            
        for node in self.G.nodes():
            self.get_community(node)
        
    
    def get_community(self, node):
        comm = node
        for node2comm in self.node2comm_list:
            comm = node2comm(comm)
        self.G._node[node].update({'category':comm})

### end of TODO ###


def main():

    G = getGraph()
    
    ### TODO ###
    # implement your community detection alg. here
    LouvainClass = Louvain(G=G)
    LouvainClass.louvain()
    G = LouvainClass.G
    ### end of TODO ###

    store_result(G)

if __name__ == "__main__":
    main()