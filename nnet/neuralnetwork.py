import numpy as np
import scipy as sp
import cPickle, os
from .layers import ParamMixin
from .helpers import one_hot, unhot
import enum_local as LOAD

class NeuralNetwork:
    def __init__(self, layers, rng=None):
        self.layers = layers
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

    def _setup(self, X, Y):
        # Setup layers sequentially
        next_shape = X.shape
        for layer in self.layers:
            layer._setup(next_shape, self.rng)
            next_shape = layer.output_shape(next_shape)
#            print(next_shape)
        if next_shape != Y.shape:
            raise ValueError('Output shape %s does not match Y %s'
                             % (next_shape, Y.shape))

    def fit(self, X, Y, learning_rate=0.1, max_iter=10, batch_size=64, name="mnist", load_type = LOAD.NUMERIC):
        """ Train network on the given data. """
        n_samples = Y.shape[0]
        n_batches = n_samples // batch_size
        Y_one_hot = one_hot(Y , load_type=load_type)
        self._setup(X, Y_one_hot)
        self.load_file(name=name)
        iter = 0
        # Stochastic gradient descent with mini-batches
        while iter < max_iter:
            iter += 1
            for b in range(n_batches):
                print (str(b + 1) + " of " + str(n_batches) +" batches, iter " 
                    + str(iter) + " with total of "+ str(max_iter))
                batch_begin = b*batch_size
                batch_end = batch_begin+batch_size
                X_batch = X[batch_begin:batch_end]
                Y_batch = Y_one_hot[batch_begin:batch_end]

                # Forward propagation
                X_next = X_batch
                for layer in self.layers:
                    X_next = layer.fprop(X_next)
                Y_pred = X_next

                # Back propagation of partial derivatives
                next_grad = self.layers[-1].input_grad(Y_batch, Y_pred)
                for layer in reversed(self.layers[:-1]):
                    next_grad = layer.bprop(next_grad)

                # Update parameters
                for layer in self.layers:
                    if isinstance(layer, ParamMixin):
                        for param, inc in zip(layer.params(),
                                              layer.param_incs()):
                            param -= learning_rate*inc

            # Output training status
            print("\nfind loss and error")
            loss = self._loss(X, Y_one_hot)
            error = self.error(X, Y)
            print('iter %i, loss %.4f, train error %.4f' % (iter, loss, error))
            self.save_file(name=name)

    def _loss(self, X, Y_one_hot):
        X_next = X
        for layer in self.layers:
            X_next = layer.fprop(X_next)
        Y_pred = X_next
        return self.layers[-1].loss(Y_one_hot, Y_pred)

    def predict(self, X):
        """ Calculate an output Y for the given input X. """
        X_next = X
        for layer in self.layers:
            X_next = layer.fprop(X_next)
        Y_pred = unhot(X_next)
        return Y_pred

    def error(self, X, Y):
        """ Calculate error on the given data. """
        Y_pred = self.predict(X)
        error = Y_pred != Y
        return np.mean(error)

    def check_gradients(self, X, Y):
        """ Helper function to test the parameter gradients for
        correctness. """
        # Warning: the following is a hack
        print "check gradients..."
        Y_one_hot = one_hot(Y) ## this line needs a load_type!!
        self._setup(X, Y_one_hot)
        for l, layer in enumerate(self.layers):
            if isinstance(layer, ParamMixin):
                print('layer %d' % l)
                for p, param in enumerate(layer.params()):
                    param_shape = param.shape

                    def fun(param_new):
                        param[:] = np.reshape(param_new, param_shape)
                        return self._loss(X, Y_one_hot)

                    def grad_fun(param_new):
                        param[:] = np.reshape(param_new, param_shape)
                        # Forward propagation
                        X_next = X
                        for layer in self.layers:
                            X_next = layer.fprop(X_next)
                        Y_pred = X_next

                        # Back-propagation of partial derivatives
                        next_grad = self.layers[-1].input_grad(Y_one_hot,
                                                               Y_pred)
                        for layer in reversed(self.layers[l:-1]):
                            next_grad = layer.bprop(next_grad)
                        return np.ravel(self.layers[l].param_grads()[p])

                    param_init = np.ravel(np.copy(param))
                    err = sp.optimize.check_grad(fun, grad_fun, param_init)
                    print('diff %.2e' % err)
                    
    def save_file(self, name = "mnist"):
        print ("saving " + name)
        W = []
        b = []
        level = 0
        for layer in self.layers:
            level += 1
            if isinstance(layer, ParamMixin):
                W, b = layer.params()
                # pickle W and b
                f1 = file(str('../nn/'+name+'-weights'+ str(level) +'.save'), 'wb')
                cPickle.dump(W, f1, protocol=cPickle.HIGHEST_PROTOCOL)
                f1.close()
                f2 = file(str('../nn/'+name+'-bias'+ str(level) +'.save'), 'wb')
                cPickle.dump(b, f2, protocol=cPickle.HIGHEST_PROTOCOL)
                f2.close()
                if False:
                    # pickle W and b
                    dW, db = layer.param_incs()
                    f3 = file(str('../nn/'+name+'-d-weights'+ str(level) +'.save'), 'wb')
                    cPickle.dump(dW, f3, protocol=cPickle.HIGHEST_PROTOCOL)
                    f3.close()
                    f4 = file(str('../nn/'+name+'-d-bias'+ str(level) +'.save'), 'wb')
                    cPickle.dump(db, f4, protocol=cPickle.HIGHEST_PROTOCOL)
                    f4.close()
        
        
    def load_file(self, name = "mnist"):
        #print len(self.layers)
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], ParamMixin):
                path1 = str("../nn/"+name+"-weights" + str(i+1) + ".save")
                if os.path.exists(path1):
                    f1 = file(path1, 'rb')
                    loaded_obj1 = cPickle.load(f1)
                    f1.close()
                    self.layers[i].W = loaded_obj1
                    print ("load " + path1)
                path2 = str("../nn/"+name+"-bias" + str(i+1) + ".save")
                if os.path.exists(path2):
                    f2 = file(path2, 'rb')
                    loaded_obj2 = cPickle.load(f2)
                    f2.close()
                    self.layers[i].b = loaded_obj2
                    print ("load " + path2)
                if False:
                    path3 = str("../nn/"+name+"-d-weights" + str(i+1) + ".save")
                    if os.path.exists(path3):
                        f3 = file(path3, 'rb')
                        loaded_obj3 = cPickle.load(f3)
                        f3.close()
                        self.layers[i].dW = loaded_obj3
                        print ("load " + path3)
                    path4 = str("../nn/"+name+"-d-bias" + str(i+1) + ".save")
                    if os.path.exists(path4):
                        f4 = file(path4, 'rb')
                        loaded_obj4 = cPickle.load(f4)
                        f4.close()
                        self.layers[i].db = loaded_obj4
                        print ("load " + path4)
            
    def append_status(self, name, message):
        filename = "status-" + name.strip() +".txt"
        f = open(filename, 'a')
        f.write(message)
        print (message)
        f.close()
