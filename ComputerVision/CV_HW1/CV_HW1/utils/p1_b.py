import cv2
import numpy as np
import sys

class Union_Find:
    def __init__(self, n):
        self.uf = [-1] * (n+1)
        self.num_sets = n

    def find(self, p):
        r = p
        while self.uf[p] > 0:
            p = self.uf[p]
        while r != p:
            self.uf[r], r =p, self.uf[r]
        return p
    
    def union(self, p, q):
        root_p = self.find(p)
        root_q = self.find(q)
        if root_p == root_q :
            return
        elif self.uf[root_p] > self.uf[root_q]:
            self.uf[root_q] += self.uf[root_p]
            self.uf[root_p] = root_q
        else:
            self.uf[root_p] += self.uf[root_q]
            self.uf[root_q] = root_p
        self.num_sets -= 1

    def if_connected(self , p, q):
        return self.find(p) == self.find(q)

def padding (binary_image):
    return np.pad(binary_image, ((1,1), (1,1)), 'constant', constant_values=(0,0))
    
def first_pass (binary_image, uf_set):
    offsets = [[-1,-1], [-1,0], [0,-1]]
    hei, wid = binary_image.shape[0], binary_image.shape[1]
    first_image = binary_image
    lab = 1
    
    for y in range(hei):
        for x in range(wid):
            if first_image[y][x] == 0:
                continue

            neighbor = []
            for offset in offsets:
                if first_image[y+offset[0], x+offset[1]] != 0:
                    neighbor.append(first_image[y+offset[0], x+offset[1]])

            neighbor_unique = np.unique(neighbor)

            if len(neighbor_unique) == 0:
                first_image[y][x] = lab
                lab += 1
            elif len(neighbor_unique) == 1:
                first_image[y][x] = neighbor_unique[0]
            else:
                first_image[y][x] = neighbor_unique[0]
                for n in neighbor_unique:
                    uf_set.union(neighbor_unique[0], n)
    
    return first_image

            

            
def second_pass (first_image, uf_set):
    hei, wid = first_image.shape[0], first_image.shape[1]
    second_image = first_image
    for x in range(wid):
        for y in range(hei):
            if second_image[hei-y-1][x] != 0:
                second_image[hei-y-1][x] = uf_set.find(second_image[hei-y-1][x])
    return second_image