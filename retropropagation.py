# Back propagation algorithm for learning in multilayered networks

import math
from collections import defaultdict

def g(val):
    return 1.0 / (1.0 + math.exp(-val))

def gPrime(val):
    return g(val) * (1.0 - g(val))

class BackPropNeuralNetwork():
    def __init__(self, links):
        '''Creates a network based on the given topology.
        links : [(node1, node2), ..., (node3, node4)]'''
        self.weight = defaultdict(dict)  # weights between two nodes
        self.children = defaultdict(list)

        for link in links:
            self.__addSynapse(*link)
            
        self.layer = {}
        self.__initLayers()
        
    def __addSynapse(self, fromNeurone, toNeurone, intialWeight=0.5):
        '''Adds a directional link between two nodes/neurons'''
        self.weight[fromNeurone][toNeurone] = intialWeight
        self.children[toNeurone].append(fromNeurone)

    def backPropagationLearning(self, examples, alpha):
        '''Do one iteration of the retropropagation algorithm
        with all the given examples'''
        
        # For (input, output) in given examples
        for activation, expectedOutput in examples:
            In = {}
            for level in xrange(2, self.nbLayers + 1):
                for i in self.layer[level]:
                    In[i] = sum(self.weight[j][i] * activation[j] for j in self.children[i])
                    activation[i] = g(In[i])
            
            delta = {}
            # Calculate delta of output layer
            for i in self.layer[self.nbLayers]:
                delta[i] = gPrime(In[i]) * (expectedOutput[i] - activation[i])
            
            # Propagate deltas backward in intermediate layers
            for level in xrange(self.nbLayers - 1, 1, -1):
                # For nodes in current layer
                for j in self.layer[level]:  # j is current node
                    # Calculate the error based on weights to higher layers
                    delta[j] = gPrime(In[j]) * sum(self.weight[j][i] * delta[i] for i in self.weight[j])
            
            # Update every weight in the network using deltas
            for j in self.weight:
                for i in self.weight[j]:
                    self.weight[j][i] += alpha * activation[j] * delta[i]

        # print activation
        # print delta
        # print self.weight
    
    def __initLayers(self):
        '''Finds the neurones per layer 
        and the number of layers of the graph'''
        fromN = set()
        toN = set()
        for n1, n2Weight in self.weight.items():
            fromN.add(n1)
            toN.update(n2Weight.keys())
                
        outputNodes = toN - fromN  # Top layer
        inputNodes = fromN - toN  # bottom layer
        
        level = 1  # Level of bottom layer
        levelNodes = inputNodes
        while len(levelNodes):
            self.layer[level] = levelNodes
            level += 1
            levelNodes = set(n2 for n1 in levelNodes
                             for n2 in self.weight[n1]
                             if n2 not in outputNodes)

        self.layer[level] = outputNodes
        self.nbLayers = len(self.layer)
        print self.layer, self.nbLayers

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
examples = [({0:0.8, 1:0.5, 2:0.5, 3:0.4}, {6:0.2, 7:0.5})]  # list of (inputs, output)
alpha = 1.0

r = BackPropNeuralNetwork(links)
r.backPropagationLearning(examples, alpha)
#print examples

print "New synapse weights:"
for n1, n2Weight in r.weight.items():
    if len(n2Weight):
        for n2, weight in n2Weight.items():
            print '{} -> {} = {:0.3}'.format(n1, n2, weight)
