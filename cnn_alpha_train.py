#!/usr/bin/env python
# coding: utf-8

import time, signal, sys
import numpy as np
import sklearn.datasets
import nnet.neuralnetwork as cnnet
import nnet.convnet.layers as conv
import nnet.layers as lnnet
import enum_local as LOAD
import load_png_alpha as lp
import datetime

def run():
    name = "alpha"
    print str(datetime.datetime.now())
    
    def signal_handler(signal, frame) :
        print(" you want to exit!")
        for layer in nn.layers:
            if (isinstance(layer, lnnet.ParamMixin)) : print "len: " + str(len(layer.W))
        if True  :
            print("save weights.")
            sys.stdout.flush()
            nn.save_file(name=name)
        sys.exit(0)

    signal.signal(signal.SIGINT,signal_handler)

    n_train_samples = 300 #3000
    # Fetch data
    
    
    dset = lp.get_dataset(load_type=LOAD.ALPHA)
    #t1, l1, files = lp.batch_load_alpha( 0,  n_train_samples, True, [], LOAD.ALPHA)
    X_train = dset[0]
    y_train = dset[1]
    
    X_train = np.reshape(X_train, (-1, 1, 28, 28))
    y_train = np.array(y_train)
    
    train_idxs = np.random.random_integers(0, len(dset[0])-1, n_train_samples)
    #train_idxs = np.array([i for i in range(n_train_samples)])
    X_train = X_train[train_idxs, ...]
    y_train = y_train[train_idxs, ...]
    
    
    # Downsample training data

    
    n_classes = len(lp.ascii_ymatrix(LOAD.ALPHA))

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

    stamp = str("stamp -- "+str(datetime.datetime.now()))
    nn.append_status(name=name, message=stamp)
    # Train neural network
    t0 = time.time()
    nn.fit(X_train, y_train, learning_rate=0.05, max_iter=5, batch_size=64, name=name, load_type = LOAD.ALPHA)
    t1 = time.time()
    print('Duration: %.1fs' % (t1-t0))

    if False:
        # Evaluate on test data
        error = nn.error(X_test, y_test)
        print('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run()
