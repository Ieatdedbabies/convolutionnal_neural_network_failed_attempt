###flat_layer####
import numpy as np
import random
import ast
from scipy.special import expit

def reverse_pooling(x):
    out = np.zeros_like(x)
    idx  = x.argmax()
    out.flat[idx] = x.flat[idx]
    return out

def sigmoid(x):
    return expit(x)
##    return 1 / (1 + np.exp(-x))

def sigmoidprime(x):
    y = (0.5/np.cosh(0.5*x))**2
    return y

##def sigmoidprime(x):
##    return np.exp(-x) / ((1 + np.exp(-x))**2)

class Neural_Net():
    def __init__(self, data_input, trng_output):
        self.data_input = data_input
        self.trng_output = trng_output
        self.bias = 0
        self.nodes = np.array([len(self.data_input[0]), 128, 64, len(self.trng_output[0])])
        self.LR = 0.01
        self.weightinit()

    def randomweight(self, n):
        output = []
        pos_neg = [-1, 1]
        for i in xrange(n):
            output.append(random.random()*random.choice(pos_neg))
        return output
        
    def weightinit(self):
        self.weights = []
        for n in xrange(len(self.nodes)-1):
            temp = []
            for _ in xrange(self.nodes[n]+self.bias):
                temp.append(self.randomweight(self.nodes[n+1]))
            self.weights.append(temp)
        self.weights = [np.array(tuple(self.weights[i])) for i in xrange(len(self.weights))]
        
    
    def forward(self, data):
        self.Z = []
        self.A = [data]
        try:
            temp = self.A[0].tolist()
        except Exception:
            temp = self.A[0]
        for i in xrange(len(temp)):
            if self.bias:
                temp[i].append(1.)
        self.A = [np.array(temp)]
        
        for layer in xrange(len(self.weights)):
            self.Z.append(np.dot(temp, self.weights[layer]))
            temp = sigmoid(self.Z[layer]).tolist()
            for i in xrange(len(temp)):
                if layer != len(self.weights)-1 and self.bias:
                    temp[i].append(1.)
            self.A.append(np.array(temp))
                    
        self.output = self.A[-1]
        return self.output

    def costFunction(self):
        self.forward(self.data_input)
        self.totalcost = 0.5*sum((self.trng_output-self.output)**2)
        return self.totalcost

    def costFunctionPrime(self):
        self.forward(self.data_input)
        self.delta = [[] for x in xrange(len(self.weights))]
        self.DcostDw = [[] for x in xrange(len(self.weights))]

        for layer in reversed(xrange(len(self.weights))):
            Zprime = sigmoidprime(self.Z[layer])
            if layer == len(self.weights)-1:
                self.delta[layer] = np.multiply(-(self.trng_output-self.A[-1]), Zprime)
            else:
                self.delta[layer] = np.dot(self.delta[layer+1], self.weights[layer+1].T) * Zprime
            self.DcostDw[layer] = np.dot(self.A[layer].T, self.delta[layer])
        
        return self.DcostDw

    def backprop(self, LR):
        self.DcostDw = (np.array(self.DcostDw)*LR).tolist()
        self.weights = (np.array(self.weights) - np.array(self.DcostDw)).tolist()

    def training(self, iteration, LR):
        for i in xrange(iteration):
            self.costFunctionPrime()
            self.backprop(LR)
            if (i/1000.0) == (i/1000):
                print self.costFunction()
        print sum(self.costFunction())/len(self.costFunction())

def generate_dataset(output_dim = 8,num_examples=1000):
    def int2vec(x,dim=output_dim):
        out = np.zeros(dim)
        binrep = np.array(list(np.binary_repr(x))).astype('int')
        out[-len(binrep):] = binrep
        return out

    x_left_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    x_right_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    y_int = x_left_int + x_right_int

    x = list()
    for i in range(len(x_left_int)):
        x.append(np.concatenate((int2vec(x_left_int[i]),int2vec(x_right_int[i]))))

    y = list()
    for i in range(len(y_int)):
        y.append(int2vec(y_int[i]))

    x = np.array(x)
    y = np.array(y)
    
    return (x,y)

##binary_data = generate_dataset(num_examples=500)
##        
##DATA_input, trng_output = binary_data[0], binary_data[1]
##
##NN = Neural_Net(DATA_input, trng_output)
##
##NN.training(2, NN.LR)
