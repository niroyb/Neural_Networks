# Perceptron

import math
from copy import deepcopy
from collections import defaultdict


def g(val):
    return 1.0 / (1.0 + math.exp(-val))

def gPrime(val):
    return g(val) * (1.0 - g(val))

class ReseauNeurones():
    def __init__(self):
        self.graph = defaultdict(dict)
        self.back = defaultdict(list) #Noeuds fils
        
    def addSynapse(self, fromNeurone, toNeurone, intialWeight=0.5):
        '''Adds a link between two neurones'''
        self.graph[fromNeurone][toNeurone] = intialWeight
        self.back[toNeurone].append(fromNeurone)

    def epoque(self, examples, alpha):
        '''Do one iteration of the retropropagation algorithm
        with all the given examples'''
        self.__initLayers()
        
        # For input, output in given examples
        for A, y in examples:
            In = deepcopy(A)
            for L in xrange(2, self.M + 1):
                for i in self.couche[L]:
                    In[i] = sum(self.graph[j][i]*A[j] for j in self.back[i])
                    A[i] = g(In[i])
            
            delta = {}
            # Calculate error on expected output
            for i in self.top:
                delta[i] = gPrime(In[i])*(y[i] - A[i])
                
            newGraph = deepcopy(self.graph)
            
            # Propagate error correction to layers below
            
            #For all layers but the top one
            for L in xrange(self.M-1, 0, -1):
                # For nodes in current layer
                for j in self.couche[L]: # j is current node
                    # Calculate the error based on weights to higher layers
                    delta[j] = gPrime(In[j])*sum(self.graph[j][i]*delta[i] for i in self.graph[j])
                    # Assign corrected weights for links with higher layer
                    for i in self.graph[j]: # i is a parent of j
                        newGraph[j][i] += alpha*A[j]*delta[i]
            self.graph = newGraph
        #print A
        #print delta
        #print self.graph
    
    def __initLayers(self):
        '''Finds the neurones per layer and the number of layers'''
        fromN = set()
        toN = set()
        for n1 in self.graph:
            fromN.add(n1)
            for n2 in self.graph[n1]:
                toN.add(n2)
        self.top = toN - fromN #Top layer
        # mid = fromN & toN #Hidden neurons
        bot = fromN - toN  # Entry layer
        
        self.couche = {}
        level = 1
        levelNodes = bot
        while len(levelNodes):
            self.couche[level] = levelNodes
            levelNodes = set(n2 for n1 in levelNodes for n2 in self.graph[n1])
            level += 1
        self.M = len(self.couche)
        print self.couche, self.M

links = [(0 , 4, 1),
         (1, 4, 0.5),
         (1, 5, 0.6),
         (2, 4, 0.8),
         (2, 5, 1),
         (3, 5, 0.2),
         (4, 6, 0.1),
         (4, 7, 1),
         (5, 6, 0.5),
         (5, 7, 1)]

r = ReseauNeurones()
for link in links:
    r.addSynapse(*link)

exemples = [({0:0.8, 1:0.5, 2:0.5, 3:0.4}, {6:0.2, 7:0.5})]  # list of (inputs, output)
alpha = 1.0

r.epoque(exemples, alpha)

print "New synapse weights:"
for n1, n2W in r.graph.items():
    if len(n2W):
        for n2, W in n2W.items():
            print '{} -> {} = {:0.3}'.format(n1, n2, W)
