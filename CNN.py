"""

FIRST ATTEMPT EVER AT DOING CNN, THIS IS FOR REVIEW PURPOSES AS THE CODE ISN'T PRATICALLY USABLE

"""

import numpy as np
import random
import ast
import time
from sys import getsizeof
# This was for debugging purposes
np.set_printoptions(threshold=np.nan)

# Activation function I used
def relu(x):
    return np.maximum(0, x)

def reluprime(x):
    return np.where(x > 0, 1.0, 0.0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidprime(x):
    return np.exp(-x) / ((1 + np.exp(-x))**2)

# Attempt at dot multiplication with 3d array. Unused
def dot(a, b):
    if len(a[0]) != len(b):
        raise ValueError("Wrong Dimension")

    temp = [[np.array([0]*len(a[0][0])) for _ in xrange(len(b[0]))] for _ in xrange(len(a))]
    for i in xrange(len(a)):
        for x in xrange(len(b[0])):
            for j in xrange(len(b)):
                temp[i][x] += np.array(a[i][j])*b[j][x]
    return temp


## Raw data of 20 images, 10 dogs and 10 cats. Encoded with Image_to_input.py
with open("img_data.txt", "r") as f:
    trng_input = ast.literal_eval(f.readline().rstrip())

##with open("trng_output.txt", "r") as f:
##    trng_output = ast.literal_eval(f.readline().rstrip())

## One hot type of output (for cats and dogs in this attempt)
trng_output = [[1, 0], [0, 1]]

trng_input = np.array(tuple(trng_input), dtype=float)
trng_output = np.array(tuple(trng_output), dtype=float)



class Convolutional_Neural_Net():
    def __init__(self, data_input):
        self.data_input = data_input
        # Number of kernel for each layer
        self.kernel = (20, 15, 10)
        # Pooling size, the 2 first digits are the window size, the 3rd is the sliding pixel number.
        self.Psize = (3,3,2)
        # Kernel size, the 2 first digits are the window size, the 3rd is the sliding pixel number.
        self.Ksize = (3,3,2)
        # Learning rate
        self.LR = 0.001
        self.kernelinit()

    def randomweight(self, n):
        # Creating kernel with positive and negative number.
        output = []
        pos_neg = [-1, 1]
        for i in xrange(n):
            output.append(random.random()*random.choice(pos_neg))
        return output

    def kernelinit(self):
        # Initializing all the kernels.
        self.Kweights = []
        for n in xrange(sum(self.kernel)):
            temp = []
            for y in xrange(self.Ksize[0]):
                temp.append(self.randomweight(self.Ksize[1]))
            self.Kweights.append(temp)
        self.Kweights = [np.array(tuple(self.Kweights[i])) for i in xrange(len(self.Kweights))]
        self.Kweights = [self.Kweights[sum(self.kernel[:i]):sum(self.kernel[:i])+self.kernel[i]] for i in xrange(len(self.kernel))]
    
    def convolution(self, Kindex, img):
        # Forward convolution of all the input data
        convolutionned = []
        for i in xrange(len(img)):
            for K in xrange(self.kernel[Kindex]):
                output = [[] for _ in xrange(0, img[i].shape[0], self.Ksize[2])]
                for row in xrange(0, img[i].shape[0], self.Ksize[2]):
                    for col in xrange(0, img[i].shape[1], self.Ksize[2]):
                        brief = img[i][row:row+self.Ksize[0], col:col+self.Ksize[1]]
                        brief = np.multiply(brief, self.Kweights[Kindex][K][0:brief.shape[0], 0:brief.shape[1]])
                        output[row/self.Ksize[2]].append(np.sum(brief))
                convolutionned.append(np.array(output))

        return convolutionned

    def activation(self, img):
        # Activation function
        activated = []
        for i in xrange(len(img)):
            activated.append(relu(img[i]))

        return activated

    def activation_prime(self, img):
        # Activation function derivative for backpropagation
        activated = []
        for i in xrange(len(img)):
            activated.append(reluprime(img[i]))

        return activated

    def pooling(self, img):
        # Forward pooling of all the input data
        pooled = []
        for i in xrange(len(img)):
            output = [[] for _ in xrange(0, img[i].shape[0], self.Psize[2])]
            for row in xrange(0, img[i].shape[0], self.Psize[2]):
                for col in xrange(0, img[i].shape[1], self.Psize[2]):
                    brief = np.max(img[i][row:min(row+self.Psize[0], img[i].shape[0]), col:min(col+self.Psize[1], img[i].shape[1])])
                    output[row/self.Psize[2]].append(brief)
            pooled.append(np.array(output))

        return pooled

    def reduction(self):
        # Manual forward propagation
        self.img_num = len(self.data_input)
        self.C = []
        self.A = []
        self.P = []
        self.Aprime = []
        ### 0 ###
        self.C.append([self.convolution(0, [self.data_input[i]]) for i in xrange(self.img_num)])
        self.A.append([self.activation(self.C[0][i]) for i in xrange(self.img_num)])
        self.Aprime.append([self.activation_prime(self.C[0][i]) for i in xrange(self.img_num)])
        ### 1 ###
        self.C.append([self.convolution(1, self.A[0][i]) for i in xrange(self.img_num)])
        self.P.append([self.pooling(self.C[1][i]) for i in xrange(self.img_num)])
        self.A.append([self.activation(self.P[0][i]) for i in xrange(self.img_num)])
        self.Aprime.append([self.activation_prime(self.P[0][i]) for i in xrange(self.img_num)])
        ### 2 ###
        self.C.append([self.convolution(2, self.A[1][i]) for i in xrange(self.img_num)])
        self.P.append([self.pooling(self.C[2][i]) for i in xrange(self.img_num)])
        self.A.append([self.activation(self.P[1][i]) for i in xrange(self.img_num)])
        self.Aprime.append([self.activation_prime(self.P[1][i]) for i in xrange(self.img_num)])
        ### 1D array ###
        self.cleaned = np.array([np.array(self.A[-1][i]).flatten().tolist() for i in xrange(self.img_num)])

##a = [CNN.A[-1][0][0].flatten().tolist().index(x) for x in CNN.P[-1][0][0].flatten()]
##a = [[i/CNN.A[-1][0][0].shape[0], i%CNN.A[-1][0][0].shape[0]] for i in a]
##np.where(CNN.C[-1][0][0][:,:,None]==CNN.P[-1][0][0][CNN.P[-1][0][0]!=0])[:-1]

    def costFunctionPrime(self):
        self.delta = [[] for _ in xrange(len(self.kernel))]
        self.DcostDw = [[] for _ in xrange(len(self.kernel))]
        
        shape = np.array(self.Aprime[-1]).shape
        pooled_D = np.dot(NN.delta[0], NN.weights[0].T).reshape(shape) * self.Aprime[2]
        self.delta[2] = [self.reverse_pooling(self.P[1][img], self.C[2][img], pooled_D[img]) for img in xrange(self.img_num)]
        self.DcostDw[2] = self.cost_derivative(2, self.A[1])
        
        pooled_D = np.array([self.delta_calc(2, i, self.P[0]) for i in xrange(self.img_num)]) * self.Aprime[1]
        self.delta[1] = [self.reverse_pooling(self.P[0][img], self.C[1][img], pooled_D[img]) for img in xrange(self.img_num)]
        self.DcostDw[1] = self.cost_derivative(1, self.A[0])
        
        self.delta[0] = np.array([self.delta_calc(1, i, self.C[0]) for i in xrange(self.img_num)])*self.Aprime[0]
        temp = [[self.data_input[i]] for i in xrange(len(self.data_input))]
        self.DcostDw[0] = self.cost_derivative(0, temp)

    def cost_derivative(self, layer, a_layer):
        output = np.array(self.Kweights[layer])*0
        l_ammount = len(a_layer[0])
        for img in xrange(self.img_num):
            for F in xrange(self.kernel[layer]):
                for i in xrange(l_ammount):
                    for row in xrange(len(self.delta[layer][img][i+F*l_ammount])):
                        for col in xrange(len(self.delta[layer][img][i+F*l_ammount][row])):
                            if self.delta[layer][img][i+F*l_ammount][row][col] == 0:
                                continue
                            ROW = row*self.Ksize[2]
                            COL = col*self.Ksize[2]
                            temp = self.delta[layer][img][i+F*l_ammount][row][col] * a_layer[img][i][ROW:ROW+self.Ksize[0], COL:COL+self.Ksize[1]]
                            output[F][0:temp.shape[0], 0:temp.shape[1]] += temp
                    
        return output

    def delta_calc(self, layer, img, P_C):
        output = [P_C[img][i]*0 for i in xrange(len(P_C[img]))]
        l_ammount = len(P_C[img])
        for F in xrange(self.kernel[layer]):
            for i in xrange(l_ammount):
                for row in xrange(len(self.delta[layer][img][i+F*l_ammount])):
                    for col in xrange(len(self.delta[layer][img][i+F*l_ammount][row])):
                        if self.delta[layer][img][i+F*l_ammount][row][col] == 0:
                            continue
                        ROW = row*self.Ksize[2]
                        COL = col*self.Ksize[2]
                        back_p = self.Kweights[layer][F]*self.delta[layer][img][i+F*l_ammount][row][col]
                        temp = output[i][ROW:ROW+self.Ksize[0], COL:COL+self.Ksize[1]]
                        temp += back_p[0:temp.shape[0], 0:temp.shape[1]]
                        
        return output
        
    def reverse_pooling(self, P_layer, C_layer, D_layer): #P_layer=self.P[-1][0], C_layer=self.C[-1][0]
        output = []
        for P_index in xrange(len(P_layer)):
            C_null = C_layer[P_index]*0
            for row in xrange(len(P_layer[P_index])):
                for col in xrange(len(P_layer[P_index][row])):
                    if P_layer[P_index][row][col] == 0:
                        continue
                    C_col = col*self.Psize[2]
                    C_row = row*self.Psize[2]
                    C_matrice = C_layer[P_index][C_row:C_row+self.Psize[0],C_col:C_col+self.Psize[1]]
                    condition = np.isin(C_matrice, P_layer[P_index][row][col])
                    location = np.where(condition)
                    C_null[location[0][0]+C_row][location[1][0]+C_col] += D_layer[P_index][row][col]
            output.append(C_null)

        return np.array(output)

    def backprop(self):
        self.Kweights = (np.array(self.Kweights) - np.array(self.DcostDw)*self.LR).tolist()

class Neural_Net():
    def __init__(self, data_input):
        self.data_input = data_input
        self.bias = 0
        self.nodes = np.array([len(self.data_input[0]), 200, len(trng_output[0])])
        self.regulation = 0.000#1
        self.dividor = self.data_input.shape[0]
        if not self.regulation:
            self.dividor /= self.data_input.shape[0]
        self.LR = 0.001
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
            temp = relu(self.Z[layer]).tolist()
            for i in xrange(len(temp)):
                if layer != len(self.weights)-1 and self.bias:
                    temp[i].append(1.)
            self.A.append(np.array(temp))
                    
        self.output = self.A[-1]
        return self.output

    def costFunction(self):
        self.forward(self.data_input)
        temp = 0
        for layer in xrange(len(self.weights)):
            temp += np.sum(self.weights[layer]**2)
        temp *= (self.regulation/2)
        self.totalcost = 0.5*sum((trng_output-self.output)**2)/self.dividor + temp
        return self.totalcost

    def costFunctionPrime(self):
        self.forward(self.data_input)
        self.delta = [[] for _ in xrange(len(self.weights))]
        self.DcostDw = [[] for _ in xrange(len(self.weights))]

        for layer in reversed(xrange(len(self.weights))):
            if layer == len(self.weights)-1:
                self.delta[layer] = np.multiply(-(trng_output-self.A[-1]), reluprime(self.Z[layer]))
            else:
                Zprime = reluprime(self.Z[layer])
                temp = Zprime.tolist()
                for i in xrange(len(temp)):
                    if self.bias:
                        temp[i].append(1.)
                Zprime = np.array(temp)
                if layer < len(self.weights)-2 and self.bias:
                    temp = np.delete(self.delta[layer+1], len(self.delta[layer+1][0])-1, 1)
                else:
                    temp = self.delta[layer+1]
                self.delta[layer] = np.dot(temp, self.weights[layer+1].T) * Zprime#reluprime(self.Z[layer])
            temp = np.dot(self.A[layer].T, self.delta[layer])
            if layer != len(self.weights)-1 and self.bias:
                temp = np.delete(temp, len(temp[0])-1, 1)
            self.DcostDw[layer] = temp/self.dividor + self.regulation*self.weights[layer]
        
        return self.DcostDw

    def backprop(self, LR):
        self.DcostDw = (np.array(self.DcostDw)*LR).tolist()
        self.weights = (np.array(self.weights) - np.array(self.DcostDw)).tolist()
##        for layer in reversed(xrange(len(self.weights))):
##            for nodes in xrange(len(self.weights[layer])):
##                for weight in xrange(len(self.weights[layer][nodes])):
##                    self.weights[layer][nodes][weight] -= LR * self.DcostDw[layer][nodes][weight]

    def training(self, iteration, LR):
        temp = self.costFunction()
        print temp
        for _ in xrange(iteration):
            self.costFunctionPrime()
            for _ in xrange(1):
                self.backprop(LR)
            print self.costFunction()

CNN = Convolutional_Neural_Net(trng_input)
CNN.reduction()
NN = Neural_Net(CNN.cleaned)
NN.costFunctionPrime()
CNN.costFunctionPrime()
NN.backprop(NN.LR)
CNN.backprop()
def training(iteration):
    for _ in xrange(iteration):
        CNN.reduction()
        print NN.costFunction()
        NN.costFunctionPrime()
        CNN.costFunctionPrime()
        NN.backprop(NN.LR)
        CNN.backprop()

training(10)
##CNN.reduction()
