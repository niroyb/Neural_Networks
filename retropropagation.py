# Perceptron

import math
from copy import deepcopy
from collections import defaultdict


def g(val):
    return 1.0 / (1.0 + math.exp(-val))

def gPrime(val):
    return g(val) * (1.0 - g(val))

class ReseauNeurones():
    def __init__(self, links):
        '''Creates a network based on the given topology.
        links : [(node1, node2), ..., (node3, node4)]'''
        self.weight = defaultdict(dict)  # weights between two nodes
        self.children = defaultdict(list)

        for link in links:
            self.__addSynapse(*link)
        self.__initLayers()
        
    def __addSynapse(self, fromNeurone, toNeurone, intialWeight = 0.5):
        '''Adds a directional link between two nodes/neurons'''
        self.weight[fromNeurone][toNeurone] = intialWeight
        self.children[toNeurone].append(fromNeurone)

    def backPropagationLearning(self, examples, alpha):
        '''Do one iteration of the retropropagation algorithm
        with all the given examples'''
        
        # For (input, output) in given examples
        for A, y in examples:
            In = deepcopy(A)
            for level in xrange(2, self.M + 1):
                for i in self.layer[level]:
                    In[i] = sum(self.weight[j][i] * A[j] for j in self.children[i])
                    A[i] = g(In[i])
            
            delta = {}
            # Calculate delta of output layer
            for i in self.layer[self.M]:
                delta[i] = gPrime(In[i]) * (y[i] - A[i])
                
            newGraph = deepcopy(self.weight)
            
            # Propagate deltas backward from output to input layer
            
            # For all layers but the outputNodes one
            for level in xrange(self.M - 1, 1, -1):
                # For nodes in current layer
                for j in self.layer[level]:  # j is current node
                    # Calculate the error based on weights to higher layers
                    delta[j] = gPrime(In[j]) * sum(self.weight[j][i] * delta[i] for i in self.weight[j])
            
            # Update every weight in the network using deltas
            for j in self.weight:
                for i in self.weight[j]:
                    newGraph[j][i] += alpha * A[j] * delta[i]
            self.weight = newGraph
        # print A
        # print delta
        # print self.weight
    
    def __initLayers(self):
        '''Finds the neurones per layer and the number of layers'''
        fromN = set()
        toN = set()
        for n1 in self.weight:
            fromN.add(n1)
            for n2 in self.weight[n1]:
                toN.add(n2)
        outputNodes = toN - fromN  # Top layer
        # mid = fromN & toN #Hidden neurons
        inputNodes = fromN - toN  # bottom layer
        
        self.layer = {}
        level = 1
        levelNodes = inputNodes
        while len(levelNodes):
            self.layer[level] = levelNodes
            levelNodes = set(n2 for n1 in levelNodes
                             for n2 in self.weight[n1]
                             if n2 not in outputNodes)
            level += 1

        self.layer[level] = outputNodes
        self.M = len(self.layer)
        print self.layer, self.M

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

r = ReseauNeurones(links)

exemples = [({0:0.8, 1:0.5, 2:0.5, 3:0.4}, {6:0.2, 7:0.5})]  # list of (inputs, output)
alpha = 1.0

r.backPropagationLearning(exemples, alpha)

print "New synapse weights:"
for n1, n2W in r.weight.items():
    if len(n2W):
        for n2, W in n2W.items():
            print '{} -> {} = {:0.3}'.format(n1, n2, W)
