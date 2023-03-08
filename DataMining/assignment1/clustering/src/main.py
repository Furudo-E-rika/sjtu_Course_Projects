import csv
import numpy as np
from numpy import argmax
from numpy.linalg import norm


DIMENSION = 100
TOTAL = 50000
CATEGORY = 5
THRESHOLD = 100

RANDOM_INIT_MEANS = False
# random initialing is faster but may fall into error
# while dispersed initialing costs time but has a stable performance.

data = np.zeros((TOTAL, DIMENSION))
means = np.zeros((CATEGORY, DIMENSION))
label = np.zeros(TOTAL, dtype=int)

def getData():
    global data
    with open("../data/features.csv", 'r') as csvFile:
        csvFile.readline()
        csv_reader = csv.reader(csvFile)
        PID = 0
        for row in csv_reader:
            data[PID] = row[1:] 
            PID += 1

def initRandomMeans():
    global means
    choice = np.random.choice(TOTAL, CATEGORY, replace=False)
    means = data[choice]

def initDispersedMeans():
    global means
    choice = np.zeros(CATEGORY, dtype=int)
    choice[0] = np.random.randint(TOTAL)
    distance2Means = np.full(50000, np.infty)
    for i in range(CATEGORY-1):
        for candidate in range(TOTAL):
            distance2Means[candidate] = min(distance2Means[candidate], norm(data[candidate] - data[choice[i]]))
        choice[i+1] = argmax(distance2Means)
    means = data[choice]

def initMeans():
    if RANDOM_INIT_MEANS:
        initRandomMeans()
    else:
        initDispersedMeans()

def storeResult(filename, labelMap):
    with open(filename, 'w') as output:
        output.write("id,category\n")
        for i in range(TOTAL):
            output.write("{},{}\n".format(i, labelMap[label[i]]))

def mapLabelName():
    ### return the map: current cluster ID -> radius-sorted cluster ID
    radius = np.zeros(CATEGORY)
    ### TODO: calculate radius of each cluster and store them in var:radius ###
    global data, means, label
    for i in range(len(data)):
        if (norm(data[i] - means[label[i]]) > radius[label[i]]):
            radius[label[i]] = norm(data[i] - means[label[i]])

    ### end of TODO ###
    temp =  radius.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(CATEGORY)
    print("Radius: ")
    print(radius[temp])
    return ranks

### TODO ###
### you can define some useful function here if you want
def update_means():
    ## this function is used for updating the center of each category with fixed label
    global means
    new_means = np.zeros((CATEGORY, DIMENSION))
    num = np.zeros(CATEGORY, dtype=int)
    for i in range(len(data)):
        for cate in range(CATEGORY):
            if label[i] == cate:
                num[cate] += 1
                new_means[cate] += data[i]
                break
    
    for cate in range(CATEGORY):
        if num[cate] != 0:
            new_means[cate] = new_means[cate] / num[cate]
        else:
            new_means[cate] = means[cate]
    
    means = new_means

def update_labels():
    ## this function is used for updating the label of each data with fixed category center
    global label
    new_label = np.zeros(TOTAL, dtype=int)
    distance2Means = np.full(50000, np.infty)
    for i in range(len(data)):
        for m in range(len(means)):
            if norm(data[i] - means[m]) < distance2Means[i]:
                distance2Means[i] = norm(data[i] - means[m])
                new_label[i] = m
                

    label = new_label

### end of TODO ###
        

def main():
    getData()
    initMeans()
    ### TODO ###
    # implement your clustering alg. here
    epochs = 10
    for i in range(epochs):
        update_labels()
        update_means()
        

    ### end of TODO ###
    labelMap = mapLabelName()
    storeResult("../data/predictions.csv", labelMap)


if __name__ == "__main__":
    main()
        
        