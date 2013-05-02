#Perceptron

import math

def g(val):
    return 1.0/(1.0 + math.exp(-val))

def gPrime(val):
    return g(val)*(1.0- g(val))

def perceptron(examples, weights, alpha):
    for entry, y in examples:
        inVal = sum(weights[j]*entry[j] for j in xrange(len(entry)))
        aOut = g(inVal)
        delta = gPrime(inVal)*(y - aOut)
        print inVal, aOut, delta

        #update weights
        for i in xrange(len(entry)):
            weights[i] += alpha*entry[i]*delta
        print weights
        
examples = [((0.8, 0.5, 0.5), 0.5)] #list of (inputs, output)
weights = [0.2, 0.5, 0.8]
alpha = 0.9

perceptron(examples, weights, alpha)
