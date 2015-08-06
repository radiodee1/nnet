#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import sklearn.datasets
import nnet.neuralnetwork as cnnet
import nnet.convnet.layers as conv
import nnet.layers as lnnet
import math, sys
from nnet.helpers import one_hot, unhot

def run():
    #conv.conv.print_test()
    # Fetch data
    mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home='./data')
    split = 60000
    X_train = np.reshape(mnist.data[:split], (-1, 1, 28, 28))/255.0
    y_train = mnist.target[:split]
    X_test = np.reshape(mnist.data[split:], (-1, 1, 28, 28))/255.0
    y_test = mnist.target[split:]
    n_classes = np.unique(y_train).size

    # Downsample training data
    n_train_samples = 1 #3000
    train_idxs = np.random.random_integers(0, split-1, n_train_samples)
    #train_idxs = np.array([i for i in range(n_train_samples)])
    X_train = X_train[train_idxs, ...]
    y_train = y_train[train_idxs, ...]
    name = "mnist"

    # Setup convolutional neural network
    nn = cnnet.NeuralNetwork(
        layers=[
            conv.Conv(
                n_feats=12,
                filter_shape=(5, 5),
                strides=(1, 1),
                weight_scale=0.1,
                weight_decay=0.001,
            ),
            lnnet.Activation('relu'),
            conv.Pool(
                pool_shape=(2, 2),
                strides=(2, 2),
                mode='max',
            ),
            conv.Conv(
                n_feats=16,
                filter_shape=(5, 5),
                strides=(1, 1),
                weight_scale=0.1,
                weight_decay=0.001,
            ),
            lnnet.Activation('relu'),
            conv.Flatten(),
            lnnet.Linear(
                n_out=n_classes,
                weight_scale=0.1,
                weight_decay=0.02,
            ),
            lnnet.LogRegression(),
        ],
    )

    

    Y_one_hot = numeric_ymatrix(y_train[0], n_classes) 
    Y_one_hot = np.array(Y_one_hot)
    
    X = X_train[0][0]
    X = np.reshape(X,(-1,1,28,28))
    Y = np.reshape(Y_one_hot,(1,10))
    
    nn._setup(X, Y)
    nn.load_file(name=name)
    
    ## the following two lines are just for textual display
    X_disp = shape_x(X_train[0][0])
    show_xvalues([X_disp], index=0)
    
    print "stored value: " + str( int(y_train[0]))
    print Y
    print("prediction: " + str( nn.predict(X)[0]))
    


def shape_x(x):
    xx = []
    for i in range(28):
        for j in range(28):
            if x[i][j] > 0.1 :
                xx.append(1)
            else:
                xx.append(0)
    return xx

def numeric_ymatrix(y, ln = 10):
    lst = []
    for i in range(ln):
        if i == y:
            lst.append(1)
        else:
            lst.append(0)
    return lst

def show_xvalues(xarray = [[]], index = 0):
    print ("show x values " + str(index))
    xx = xarray[index]
    ln = int(math.floor(math.sqrt(len(xx)))) 
    #print (ln)
    for x in range(1,ln):
        for y in range(1, ln):
            zzz = '#'
            #zzz =int( xx[x* ln + y])
            if int(xx[ x* ln + y]) == int( 0) : 
                zzz = '.'
            #print(zzz) 
            sys.stdout.write(zzz)
        print("");
    print ("\n===========================\n")

if __name__ == '__main__':
    run()
