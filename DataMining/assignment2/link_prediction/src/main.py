# please use node2vec algorithm to finish the link prediction task
# Do not change the code outside the TODO part

import networkx as nx
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
# you can use basic operations in networkx
# you can also import other libraries if you need

# read edges.csv and construct the graph
def get_graph():
    G = nx.DiGraph()
    with open("../data/lab2_edges.csv", 'r') as csvFile:
        csvFile.readline()
        csv_reader = csv.reader(csvFile)

        for row in csv_reader:
            source = int(row[0])
            target = int(row[1])
            G.add_edge(source, target)

    print("graph ready")
    return G

# TODO: finish the class Node2Vec
class Node2Vec:
    # you can change the parameters of each function and define other functions
    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, ):
        self.graph = graph
        self.node_size = 16863
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self._embeddings = {}


    def random_walk(self, node):
        walk = [node]
        for i in range(self.walk_length):
            neighbors = list(self.graph.neighbors(walk[-1]))
            if len(neighbors) > 0:
                probs = []
                for neighbor in neighbors:
                    if len(walk) > 1 and neighbor == walk[-2]:
                        probs.append(1 / self.p)
                    elif len(walk) > 1 and self.graph.has_edge(neighbor, walk[-2]):
                        probs.append(1)
                    else:
                        probs.append(1 / self.q)
                probs = [p / sum(probs) for p in probs]
                next_node = np.random.choice(neighbors, p=probs)
                walk.append(next_node)
            else:
                break
        return walk
    

    def train(self, embed_size):
        walks = []
        for node in tqdm(self.graph.nodes()):
            for i in range(self.num_walks):
                walk = self.random_walk(node)
                walks.append(walk)
        nodes = list(self.graph.nodes())

        window_size = 5
        learning_rate = 0.01
        W = np.random.randn(self.node_size, embed_size)
        for i in tqdm(range(len(walks))):
            walk = walks[i]
            for j in range(len(walk)):
                center_node = walk[j]
                neighbor_nodes = [word for word in walk[max(0, j-window_size):j] + walk[j+1:min(len(walk), j+window_size+1)]]
                
                # Precompute the dot product between center_node and neighbor_nodes
                dot_products = np.dot(W[center_node], W[neighbor_nodes].T)
                
                # Precompute the sigmoid function applied to the dot products
                sigmoids = 1 / (1 + np.exp(-dot_products))
                
                # Compute gradients
                gradients = sigmoids - 1
                
                # Update W for the center_node
                W[center_node] -= learning_rate * np.dot(gradients, W[neighbor_nodes])
                
                # Update W for the neighbor_nodes
                W[neighbor_nodes] -= learning_rate * np.outer(gradients, W[center_node])
                
        self.W = W
        return W
    

    # get embeddings of each node in the graph
    def get_embeddings(self):
        for node in range(self.node_size):
            
            self._embeddings[node] = self.W[node]

        return self._embeddings

    # use node embeddings and known edges to train a classifier
    def train_classifier(self):

        X = []
        Y = []

        for u in self.graph.nodes():
            num_positve = 0
            num_negative = 0
            for v in self.graph.nodes():
                if self.graph.has_edge(u, v):
                    if num_positve < 100:
                        X.append(np.concatenate((self._embeddings[u], self._embeddings[v])))
                        Y.append(1)
                        num_positve += 1
                else:
                    if num_negative < 100:
                        if u != v:
                            X.append(np.concatenate((self._embeddings[u], self._embeddings[v])))
                            Y.append(0)
                            num_negative += 1

                if num_positve >= 100 and num_negative >= 100:
                    break

        
        self._classifier = LogisticRegression(random_state=0)
        self._classifier.fit(X, Y)
        

    def predict(self, source, target):
        enc1 = self._embeddings[source]
        enc2 = self._embeddings[target]

        # use embeddings to predict links
        
        prob = self._classifier.predict([np.concatenate((enc1, enc2))]).item()

        return prob

### TODO ###
### you can define some useful functions here if you want


### end of TODO ###

def store_result(model):
    with open('../data/predictions.csv', 'w') as output:
        output.write("id,probability\n")
        with open("../data/lab2_test.csv", 'r') as csvFile:
            csvFile.readline()
            csv_reader = csv.reader(csvFile)
            for row in csv_reader:
                id = int(row[0])
                source = int(row[1])
                target = int(row[2])
                prob = model.predict(source, target)
                
                output.write("{},{:.4f}\n".format(id, prob))

def main():
    G = get_graph()

    model = Node2Vec(G, walk_length=20, num_walks=10000)

    model.train(embed_size=100)

    embeddings = model.get_embeddings()

    model.train_classifier()

    store_result(model)

if __name__ == "__main__":
    main()