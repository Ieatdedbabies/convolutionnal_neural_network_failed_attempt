### CNN_REMAKE ###
import numpy as np
import random
import ast
import time
from CNN_flat_layer import *
from sys import getsizeof
from PIL import Image
from PIL import ImageFilter
from os import path
folder = path.dirname(__file__)
img_folder = path.join(folder, "img")
np.set_printoptions(threshold=np.nan)

def relu(x):
    return np.maximum(0, x)

def reluprime(x):
    return (x>0).astype(x.dtype)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidprime(x):
    return np.exp(-x) / ((1 + np.exp(-x))**2)

with open("img_data_test.txt", "r") as f:
    trng_input = ast.literal_eval(f.readline().rstrip())

def output_init():
    temp = [[0,1] for _ in xrange(10)]
    temp1 = [[1,0] for _ in xrange(10)]
    temp.extend(temp1)
    return np.array(temp)

trng_output = output_init()

class Filter():
    def __init__(self, size):
        self.weights = []
        self.size = size
        for y in xrange(size[0]):
            self.weights.append([])
            for x in xrange(size[1]):
                self.weights[y].append(random.random()*random.choice([-1, 1]))
        self.weights = np.array(self.weights).reshape((self.size[0], self.size[1], 1))

    def convolution(self, img, stride, P=0):
        img = np.array(img)
        padding = np.zeros((img.shape[0]+2*P, img.shape[1]+2*P, img.shape[2]), dtype="float")
        padding[P:img.shape[0]+P, P:img.shape[1]+P, :] = img
        img = padding
        convolutionned = []
        for row in xrange(0, img.shape[0]-(self.size[0]-1), stride):
            my_row = []
            for col in xrange(0, img.shape[1]-(self.size[1]-1), stride):
                lense = img[row:row+self.size[0], col:col+self.size[1]]
                my_row.append(np.sum(np.multiply(lense, self.weights)))
            convolutionned.append(my_row)

        convolutionned = np.array(convolutionned)
        shape = convolutionned.shape

        return convolutionned.reshape((shape[0], shape[1], 1))

        
class Convolutional_Neural_Network():
    def __init__(self, data_input):
        self.data_input = data_input
        self.filters = (10,8,5)
        self.filter_size = [5,5]
        self.LR = 0.01
        self.weight_init()
        self.flat_layer_input = self.CNN_forward_propagation(self.data_input)
        

    def weight_init(self):
        self.filter_list = []
        for i in self.filters:
            self.filter_list.append([Filter(self.filter_size) for _ in xrange(i)])

    def full_convolution(self, layer, all_img, stride=1, P=0):
        full_output = []
        for img in all_img:
            output = self.filter_list[layer][0].convolution(img, stride, P=P)
            for i in xrange(1, self.filters[layer]):
                output = np.append(output, self.filter_list[layer][i].convolution(img, stride, P=P), axis=2)
            full_output.append(output)
        return np.array(full_output)

    def activation_function(self, img):
        return relu(img)

    def pooling(self, all_img, size):
        full_output = []
        for img in all_img:
            output = np.empty((img.shape[0]/2,img.shape[1]/2,img.shape[2]), dtype="float")
            for row in xrange(0, output.shape[0]):
                for col in xrange(0, output.shape[1]):
                    for depth in xrange(output.shape[2]):
                        lense = img[row*2:row*2+size[0], col*2:col*2+size[1], depth]
                        output[row, col, depth] = np.amax(lense)
            full_output.append(output)
        return np.array(full_output)

    def CNN_forward_propagation(self, my_data):
        self.Z = []
        self.A = [my_data]
        self.P = []
        for i in xrange(len(self.filters)):
            if not i:
                self.Z.append(self.full_convolution(i, self.A[i], stride=1, P=2))
            else:
                self.Z.append(self.full_convolution(i, self.P[i-1], stride=1, P=2))
            print self.Z[i].shape, "Z"
            self.A.append(self.activation_function(self.Z[i]))
            print self.A[i+1].shape, "A"
            self.P.append(self.pooling(self.A[i+1], (2,2)))
            print self.P[i].shape, "P"
        
        data = np.array([img.flatten() for img in self.P[-1]])
        return data

    def CNN_backpropagation(self):
        self.delta = [[] for x in xrange(len(self.filters))]
        self.DcostDw = [[] for x in xrange(len(self.filters))]
        
        flat_Zprime = reluprime(np.array([img.flatten() for img in self.Z[-1]]))
        print flat_Zprime.shape
        print np.dot(NN.delta[0], NN.weights[0].T).shape
        
##        self.delta[-1] = np.dot(NN.delta[0], NN.weights[0].T) * flat_Zprime

##        for layer in reversed(xrange(len(self.filters)-1)):
            
        

CNN = Convolutional_Neural_Network(trng_input)
NN = Neural_Net(CNN.flat_layer_input, trng_output)
NN.costFunctionPrime()
CNN.CNN_backpropagation()

